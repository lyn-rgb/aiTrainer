"""Per-rank worker for the real multi-process checks.

Launched by :mod:`tests.multirank.harness` as one OS process per rank, with
``MASTER_ADDR``/``MASTER_PORT``/``RANK``/``WORLD_SIZE`` in the environment.
Every case runs the production code path on a live Gloo process group and
returns a JSON-serialisable dict that the harness asserts on.

This module exists because the rest of ``tests/distributed`` is single-process:
those files import ``torch`` and check shapes at ``tp_size == 1``, so nothing in
the repository had ever created a second process.  The checks that were recorded
in ``docs/代码审计报告.md`` as "fixed, but unverifiable without a real
multi-rank job" are exercised here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import timedelta

CASES: dict[str, object] = {}


def case(name: str):
    def register(function):
        CASES[name] = function
        return function
    return register


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _max_abs_diff(left, right) -> float:
    import torch
    return float(torch.max(torch.abs(left - right)).item())


# --------------------------------------------------------------------------- #
# §4.10 a -- process-group creation order
# --------------------------------------------------------------------------- #
def _group_membership(dist, mapping, rank, groups) -> dict[str, list[int]]:
    """Every axis group's actual membership, gathered inside that group."""
    membership = {}
    coordinate = mapping.coordinate(rank)
    for axis in ("pp", "dp", "tp"):
        handle = groups.group(axis)
        expected = tuple(mapping.group_ranks(axis, coordinate))
        _require(dist.get_world_size(handle) == len(expected),
                 f"{axis}: world size {dist.get_world_size(handle)} != {len(expected)}")
        gathered: list[int] = [0] * len(expected)
        dist.all_gather_object(gathered, rank, group=handle)
        actual = tuple(sorted(gathered))
        _require(actual == expected, f"{axis}: members {actual} != expected {expected}")
        membership[axis] = list(actual)
    return membership


@case("groups_canonical_order")
def groups_canonical_order(rank: int, world: int, dist, options: dict) -> dict:
    """``ProcessGroups.create`` must terminate and yield correct memberships.

    The bug it guards: creating only a rank's own groups makes the per-process
    ``new_group`` counter carry different rank lists at the same store index, so
    no rank observes the expected barrier and every participant blocks for the
    full Gloo timeout.
    """
    from aitrainer.parallel.groups import group_creation_plan

    mapping = _mapping(world, options)
    groups = _create_groups(mapping)
    membership = _group_membership(dist, mapping, rank, groups)
    return {"rank": rank, "plan_len": len(group_creation_plan(mapping)),
            "membership": membership}


@case("groups_legacy_order_deadlocks")
def groups_legacy_order_deadlocks(rank: int, world: int, dist, options: dict) -> dict:
    """Negative control: the pre-fix per-rank ordering must NOT survive.

    Reproduces the removed implementation verbatim so that
    ``test_group_creation_order_is_load_bearing`` can prove the ordering fix is
    what makes group creation terminate, rather than asserting a green test with
    no counterfactual.
    """
    mapping = _mapping(world, options)
    coordinate = mapping.coordinate(rank)
    created = []
    for axis in ("pp", "dp", "tp"):
        axis_ranks = mapping.group_ranks(axis, coordinate)
        dist.new_group(list(axis_ranks))
        created.append(list(axis_ranks))
    return {"rank": rank, "legacy_calls": created,
            "note": "did not block, so the ordering fix is not load-bearing at this shape"}


def _mapping(world: int, options: dict):
    from aitrainer.topology import RankMapping
    pp_size, dp_size, tp_size = (int(part) for part in options.get("shape", "2,2,1").split(","))
    return RankMapping(world, pp_size=pp_size, dp_size=dp_size, tp_size=tp_size,
                       global_ranks=tuple(range(world)))


def _create_groups(mapping):
    from aitrainer.parallel.groups import ProcessGroups
    return ProcessGroups.create(mapping)


# --------------------------------------------------------------------------- #
# §4.10 b/c -- tensor parallelism and sequence parallelism, over DTensor
# --------------------------------------------------------------------------- #
def _tp_block(hidden: int):
    """A block whose projections the default plan names, with a norm in front.

    ``norm`` receives the replicated residual stream, which is the arrangement
    a real transformer has -- and the one that made torch's own
    ``SequenceParallel`` unusable here (it assumes its input is already a
    per-rank slice and silently mis-annotates otherwise).
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.q_proj = nn.Linear(hidden, hidden)
            self.o_proj = nn.Linear(hidden, hidden)

        def forward(self, value):
            return self.o_proj(torch.relu(self.q_proj(self.norm(value))))

    return Block()


def _full(value):
    return value.full_tensor() if hasattr(value, "full_tensor") else value


@case("tensor_parallel_matches_dense")
def tensor_parallel_matches_dense(rank: int, world: int, dist, options: dict) -> dict:
    """TP through ``parallelize`` must equal the dense model -- and stay DTensor.

    Both halves matter and they are different claims.  The numbers say the
    sharding is correct; the *parameter type* says the sharding is represented,
    which is what ``torch.distributed.checkpoint`` needs in order to see a
    global tensor instead of "whatever this rank happened to hold".  The old
    hand-written layers passed the first and failed the second, which is how
    ``save_sharded`` came to silently keep one rank's shard for everybody.

    The comparison is against a dense model built from the same seed, so this
    catches a consistently wrong layout as well as a diverging one.
    """
    import torch

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    hidden, batch, length, seed = 8, 2, 4, 11
    torch.manual_seed(seed)
    dense = _tp_block(hidden)
    generator = torch.Generator().manual_seed(4321)
    value = torch.randn(batch, length, hidden, generator=generator).requires_grad_(True)
    expected = dense(value)
    expected.square().mean().backward()

    torch.manual_seed(seed)
    block = _tp_block(hidden)
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": world}})
    runtime = Runtime(device="cpu", init_process_group=False)
    model = parallelize(block, config=config, runtime=runtime)

    inputs = value.detach().clone().requires_grad_(True)
    output = model(inputs)
    forward_error = _max_abs_diff(output, expected)
    output.square().mean().backward()

    errors = {}
    for name, mine, reference in (("q_proj", model.q_proj, dense.q_proj),
                                  ("o_proj", model.o_proj, dense.o_proj)):
        for field in ("weight", "bias"):
            got = getattr(mine, field).grad
            want = getattr(reference, field).grad
            errors[f"{name}.{field}"] = max(
                _max_abs_diff(_full(got), want), _max_abs_diff(_full(getattr(mine, field)),
                                                              getattr(reference, field)))
    return {"rank": rank, "forward_error": forward_error,
            "input_grad_error": _max_abs_diff(_full(inputs.grad), value.grad),
            "parameter_errors": errors,
            "weight_type": type(model.q_proj.weight).__name__,
            "weight_placements": str(model.q_proj.weight.placements),
            "weight_global_shape": list(model.q_proj.weight.shape),
            "weight_local_shape": list(model.q_proj.weight.to_local().shape),
            "expected_global_shape": list(dense.q_proj.weight.shape)}


@case("composition_matches_dense")
def composition_matches_dense(rank: int, world: int, dist, options: dict) -> dict:
    """Compounded axes over one stage: ``parallelize`` must still equal dense.

    This is the case that was missing, and its absence had a cost.  Once TP
    parameters became DTensors, ``fully_shard`` began rejecting the DP mesh
    ``wrap_fsdp`` had always built -- ``DeviceMesh.from_group`` makes a mesh with
    no parent, and FSDP2 requires the DP and TP meshes to share one:

        tp_size=2 + fsdp.enabled  ->  AssertionError: FSDP requires the DP and
        model parallel TP/EP mesh to have the same parent mesh

    while ``dp_size=2`` alone worked.  No test combined the axes, so nothing
    said so; ``parallel.dp_axis_mesh`` is the fix and this is the regression.

    ``options["shape"]`` names the axes, so the same case covers
    ``dp=2``, ``tp=2``, ``tp=2 + sp`` and (at world 4) ``dp=2, tp=2``.  Every
    rank receives **identical** data, which makes the single-process run a valid
    reference: a DP reduction over identical replicas is the identity on the
    gradient, so anything the sharding gets wrong shows up as a difference.
    Diverging-replica detection is a separate case
    (``trainer_fit_fsdp_accumulation``), because identical data cannot see it.
    """
    import torch

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    dp_size, tp_size, sp_backend = _shape(world, options)

    hidden, batch, length, seed = 8, 2, 4, 77
    torch.manual_seed(seed)
    dense = _tp_block(hidden)
    generator = torch.Generator().manual_seed(2024)
    value = torch.randn(batch, length, hidden, generator=generator).requires_grad_(True)
    expected = dense(value)
    expected.square().mean().backward()

    torch.manual_seed(seed)
    block = _tp_block(hidden)
    config = FrameworkConfig.from_dict({
        "parallel": {"dp_size": dp_size, "tp_size": tp_size, "sp_backend": sp_backend},
        # FSDP is enabled even at dp_size=1, which is what ``parallel_matrix``
        # and ``ConfigPreset.fsdp()`` build.  That configuration is not a no-op:
        # it is the one whose standalone DP mesh tripped FSDP2's parent-mesh
        # check, so skipping it here would skip the very case that broke.
        "fsdp": {"enabled": world > 1},
    })
    runtime = Runtime(device="cpu", init_process_group=False)
    model = parallelize(block, config=config, runtime=runtime) if world > 1 else block

    inputs = value.detach().clone().requires_grad_(True)
    output = model(inputs)
    output.square().mean().backward()
    errors = {name: _max_abs_diff(_full(getattr(model, name).weight.grad),
                                  getattr(dense, name).weight.grad)
              for name in ("q_proj", "o_proj")}
    return {"rank": rank, "world": world, "dp_size": dp_size, "tp_size": tp_size,
            "sp_backend": sp_backend,
            "forward_error": _max_abs_diff(output, expected),
            "input_grad_error": _max_abs_diff(_full(inputs.grad), value.grad),
            "parameter_errors": errors,
            "weight_type": type(model.q_proj.weight).__name__,
            "weight_placements": str(model.q_proj.weight.placements)}


def _shape(world: int, options: dict) -> tuple[int, int, str]:
    """``(dp, tp, sp)`` for this case, defaulting to a shape that fits ``world``."""
    if "dp" in options or "tp" in options:
        dp_size = int(options.get("dp", 1))
        tp_size = int(options.get("tp", 1))
    elif world == 4:
        dp_size, tp_size = 2, 2
    else:
        dp_size, tp_size = 2, 1
    _require(dp_size * tp_size == world,
             f"dp={dp_size} x tp={tp_size} does not match world={world}")
    return dp_size, tp_size, options.get("sp", "none")


@case("tensor_parallel_refuses_an_unmatched_plan")
def tensor_parallel_refuses_an_unmatched_plan(rank: int, world: int, dist, options: dict) -> dict:
    """A plan that names nothing must raise instead of replicating silently.

    Measured before the guard: ``tp_size=2`` over a plain ``nn.Sequential``
    left the parameter count unchanged and printed nothing -- every rank ran a
    full copy of the model on its own slice of the data, with no collectives
    and no error.  The failure mode is a model that trains and never converges,
    which is worse than a crash.
    """
    import torch

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": world}})
    runtime = Runtime(device="cpu", init_process_group=False)
    try:
        parallelize(model, config=config, runtime=runtime)
    except (ValueError, RuntimeError) as exc:
        # ValueError covers TPConfigurationError; RuntimeError is what torch's
        # own distributed errors are.  Catching those two rather than Exception
        # keeps a genuine bug (a NameError, say) from being reported as a
        # successful refusal.
        return {"rank": rank, "refused": True, "message": f"{type(exc).__name__}: {exc}"}
    return {"rank": rank, "refused": False, "message": ""}


@case("sequence_parallel_matches_dense")
def sequence_parallel_matches_dense(rank: int, world: int, dist, options: dict) -> dict:
    """SP inside the norms must equal the dense model, and must really shard.

    Two independent claims, and the second is the one that cannot be seen in
    the numbers:

    * numerics -- forward, the replicated norm parameters' gradients, and the
      input gradient all match a dense reference.  The input gradient is the
      sharp edge: an earlier version of the style took the local ``chunk`` by
      hand, and the chunk's backward gives each rank only its own slice's
      contribution, so a *replicated* input ends up with a partially-filled
      gradient -- measured at 4.8e-03 and 7.0e-03 against 9.3e-10.  Writing the
      scatter as a ``Replicate -> Shard`` layout transition instead lets
      DTensor's autograd supply the matching all-gather.
    * that the activation inside the norm is genuinely ``Shard(sequence_dim)``
      with a global sequence length of ``length`` and a local one of
      ``length / world``.  A style that sharded nothing would still match the
      dense reference exactly -- same numbers, no communication -- so the
      layout observation is the only thing standing between "sequence
      parallel" and "a no-op that reports success".
    """
    import torch

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    hidden, batch, length, seed = 8, 2, 6, 4242
    torch.manual_seed(seed)
    dense = _tp_block(hidden)
    generator = torch.Generator().manual_seed(99)
    value = torch.randn(batch, length, hidden, generator=generator).requires_grad_(True)
    dense(value).square().mean().backward()

    torch.manual_seed(seed)
    block = _tp_block(hidden)
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": world, "sp_backend": "megatron"}})
    runtime = Runtime(device="cpu", init_process_group=False)
    model = parallelize(block, config=config, runtime=runtime)

    observed: dict = {}

    def observe(module, inputs):
        activation = inputs[0]
        observed["type"] = type(activation).__name__
        observed["placements"] = str(getattr(activation, "placements", None))
        observed["global_shape"] = list(getattr(activation, "shape", []))
        observed["local_shape"] = (list(activation.to_local().shape)
                                   if hasattr(activation, "to_local") else None)

    model.norm.register_forward_pre_hook(observe)

    inputs = value.detach().clone().requires_grad_(True)
    output = model(inputs)
    forward_error = _max_abs_diff(output, dense(value))
    output.square().mean().backward()
    return {"rank": rank, "forward_error": forward_error,
            "input_grad_error": _max_abs_diff(_full(inputs.grad), value.grad),
            "weight_grad_error": _max_abs_diff(_full(model.norm.weight.grad),
                                               dense.norm.weight.grad),
            "bias_grad_error": _max_abs_diff(_full(model.norm.bias.grad), dense.norm.bias.grad),
            "observed": observed, "world": world, "sequence_length": length}


# §4.10 d -- Ulysses attention
# --------------------------------------------------------------------------- #
@case("ulysses_attention_equivalence")
def ulysses_attention_equivalence(rank: int, world: int, dist, options: dict) -> dict:
    """Head-parallel attention must reproduce dense SDPA exactly."""
    import torch

    from aitrainer.parallel.sp_ulysses import distributed_attention

    group = _tp_group(world, rank)
    batch, length, heads, head_dim = 2, 8, 4, 4
    torch.manual_seed(99)
    q, k, v = (torch.randn(batch, length, heads, head_dim) for _ in range(3))

    out = distributed_attention(q, k, v, group=group)
    _require(tuple(out.shape) == (batch, length, heads, head_dim),
             f"output shape {tuple(out.shape)} != {(batch, length, heads, head_dim)}")
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    return {"rank": rank, "error": _max_abs_diff(out, dense)}


@case("ulysses_seq_lens_refuses")
def ulysses_seq_lens_refuses(rank: int, world: int, dist, options: dict) -> dict:
    """The variable-length path must refuse loudly instead of approximating."""
    import torch

    from aitrainer.parallel.sp_ulysses import SPConfigurationError, distributed_attention

    group = _tp_group(world, rank)
    q, k, v = (torch.randn(2, 8, 4, 4) for _ in range(3))
    try:
        distributed_attention(q, k, v, group=group, seq_lens=[8, 6])
    except SPConfigurationError as exc:
        return {"rank": rank, "refused": True, "message": str(exc)}
    raise AssertionError("variable-length Ulysses attention did not refuse at world_size > 1")


# --------------------------------------------------------------------------- #
# §4.1 -- no_sync gradient accumulation across replicas
# --------------------------------------------------------------------------- #
@case("fsdp_cpu_wrap")
def fsdp_cpu_wrap(rank: int, world: int, dist, options: dict) -> dict:
    """``wrap_fsdp`` must shard in place, on CPU, with a CPU-resolved device.

    Two properties, from two separate defects:

    * FSDP's own device inference resolves the custom 'mps' backend on a
      CPU-only macOS host and raises ``AttributeError: Custom backend 'mps' not
      implement torch.mps.current_device``, so ``wrap_fsdp`` has to forward the
      device the runtime resolved rather than forwarding CUDA only.
    * FSDP2's ``fully_shard`` is composable and mutates in place, so there is no
      wrapper object to hand back.  The assertion is on the *type of the
      parameters* rather than the module, because returning the same module is
      only meaningful if it actually got sharded -- a ``wrap_fsdp`` that forgot
      to call ``fully_shard`` would also return the module.
    """
    import torch

    from aitrainer.parallel.fsdp import wrap_fsdp

    class _Runtime:
        world_size = world

        class state:
            device = "cpu"

    module = torch.nn.Linear(4, 4)
    wrapped = wrap_fsdp(module, runtime=_Runtime(), mesh=None, config=None)
    wrapped(torch.randn(2, 4)).sum().backward()
    return {"rank": rank, "wrapped": type(wrapped).__name__, "in_place": wrapped is module,
            "parameter_types": sorted({type(p).__name__ for p in wrapped.parameters()})}


@case("fsdp_grad_norm_is_global")
def fsdp_grad_norm_is_global(rank: int, world: int, dist, options: dict) -> dict:
    """Gradient clipping must use one global norm, not one per shard.

    ``torch.nn.utils.clip_grad_norm_`` reads like the drop-in for FSDP1's
    ``module.clip_grad_norm_`` and is not one.  DTensor *does* register
    ``linalg.vector_norm`` -- it returns a ``_NormPartial`` whose
    ``full_tensor()`` is the sum over the mesh -- but ``nn.utils`` stacks the
    per-parameter results with ``torch.stack``, which is not DTensor-aware and
    silently yields the shard-local numbers.  Measured here, at dp=2 on a
    3-layer model with rank-distinct data, against a true global norm of
    0.678: rank 0 reported 0.551 and rank 1 reported 0.395.  Nothing raises;
    the two ranks simply clip by different factors, or skip a clip that was
    required.

    The worker reports the true norm (computed from ``full_tensor()``, which is
    the ground truth and shares no code with the clipping path), what
    ``clip_grad_norm_`` returns, and the post-clip gradients so the test can
    check that the ranks agree with each other.
    """
    import torch
    from torch import nn

    from aitrainer.mesh import DeviceMeshManager
    from aitrainer.parallel.fsdp import clip_grad_norm_, wrap_fsdp

    _require(world > 1, "a shard-local norm is only distinguishable from the global one "
                        "when there is more than one shard")

    class _Runtime:
        world_size = world

        class state:
            device = "cpu"

    torch.manual_seed(11)
    module = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    mesh = DeviceMeshManager(dp_size=world, tp_size=1, pp_size=1, world_size=world,
                             device_type="cpu")
    sharded = wrap_fsdp(module, runtime=_Runtime(), mesh=mesh, config=None)

    # Rank-distinct data, so each rank's shard has a different norm and a
    # shard-local total would disagree with the global one *and* with the other
    # rank's.
    generator = torch.Generator().manual_seed(700 + rank)
    inputs = torch.randn(5, 6, generator=generator)
    targets = torch.randn(5, 4, generator=generator)

    def truth() -> float:
        return float(sum(parameter.grad.full_tensor().pow(2).sum()
                         for parameter in sharded.parameters()).sqrt())

    def full_gradients() -> list:
        return [value for parameter in sharded.parameters()
                for value in parameter.grad.full_tensor().flatten().tolist()]

    def zero_grads() -> None:
        for parameter in sharded.parameters():
            parameter.grad = None

    def backward() -> None:
        nn.functional.mse_loss(sharded(inputs), targets).backward()

    zero_grads(); backward()
    honest = truth()
    # float("inf") as the cap: measure the norm without clipping anything.
    measured = float(clip_grad_norm_(sharded, float("inf")).item())

    # Now clip for real, and see whether the two ranks land on the same point.
    zero_grads(); backward()
    cap = 0.05
    reported = float(clip_grad_norm_(sharded, cap).item())
    clipped = truth()
    gradients = full_gradients()
    return {"rank": rank, "true_norm": honest, "measured_norm": measured,
            "reported_norm": reported, "post_clip_norm": clipped, "cap": cap,
            "gradients": gradients}


@case("fsdp_accumulation_window_suppresses_reduction")
def fsdp_accumulation_window_suppresses_reduction(rank: int, world: int, dist, options: dict) -> dict:
    """The accumulation window must actually suppress the reduce-scatter.

    This is the one Stage-A property that fails silently.  FSDP1 answered
    ``no_sync()``; FSDP2 renamed it to ``set_requires_gradient_sync(bool)``, and
    ``DistributedModel.no_sync`` used to be
    ``getattr(self.module, "no_sync", nullcontext)()`` -- a missing method
    degraded to a no-op, so every microbatch would have been reduced separately.
    Nothing raises; the run is merely slower and the reduction timing differs.

    ``trainer_fit_fsdp_accumulation`` cannot catch that: both ranks there are fed
    *identical* data, so reducing per microbatch and reducing once at the
    boundary produce the same sum.  Hence rank-distinct data here, and an
    explicit assertion that reduced and local gradients differ -- with identical
    data the whole case would pass vacuously.

    The mechanism is measured, not assumed (Gloo, world=2, torch 2.9):

    * one synced backward populates ``.grad`` -- that value is ``unit``;
    * after each unsynced backward ``.grad`` is still ``None``.  FSDP2 holds the
      accumulated gradient in an internal unsharded buffer and materialises
      nothing, so an optimizer step taken mid-window would be a silent no-op;
    * the next *synced* backward flushes all four accumulations at once, giving
      exactly ``4 * unit``, not ``1 * unit``.  That is the property the whole
      accumulation design rests on, and it is not documented anywhere.
    """
    import torch
    from torch import nn

    from aitrainer.distributed_model import DistributedModel
    from aitrainer.mesh import DeviceMeshManager
    from aitrainer.parallel.fsdp import wrap_fsdp

    _require(world > 1, "an accumulation window needs replicas to reduce across")

    class _Runtime:
        world_size = world

        class state:
            device = "cpu"

    torch.manual_seed(3)
    module = nn.Linear(4, 4, bias=True)
    initial = {name: value.detach().clone() for name, value in module.named_parameters()}
    mesh = DeviceMeshManager(dp_size=world, tp_size=1, pp_size=1, world_size=world,
                             device_type="cpu")
    sharded = wrap_fsdp(module, runtime=_Runtime(), mesh=mesh, config=None)
    model = DistributedModel(sharded)

    # Rank-distinct data: that is what makes "reduced" and "local" different.
    generator = torch.Generator().manual_seed(500 + rank)
    inputs = torch.randn(3, 4, generator=generator)
    targets = torch.randn(3, 4, generator=generator)

    def local_gradient():
        """This rank's own gradients, computed with no process group involved."""
        weight = initial["weight"].clone().requires_grad_()
        bias = initial["bias"].clone().requires_grad_()
        nn.functional.mse_loss(inputs @ weight.T + bias, targets).backward()
        return weight.grad, bias.grad

    def full_grad():
        """The global weight gradient, or None if FSDP2 has not materialised it."""
        grad = sharded.weight.grad
        return None if grad is None else grad.full_tensor()

    def zero_grads() -> None:
        for parameter in sharded.parameters():
            parameter.grad = None

    zero_grads()
    nn.functional.mse_loss(model(inputs), targets).backward()
    unit = full_grad()
    _require(unit is not None, "a synced backward must materialise a gradient")
    reduced_vs_local = float((unit - local_gradient()[0]).abs().max())

    materialised = []
    zero_grads()
    for _ in range(3):
        with model.no_sync():
            nn.functional.mse_loss(model(inputs), targets).backward()
        materialised.append(full_grad() is not None)

    nn.functional.mse_loss(model(inputs), targets).backward()  # window boundary
    total = full_grad()
    _require(total is not None, "the boundary backward must materialise a gradient")
    return {"rank": rank,
            "reduced_vs_local": reduced_vs_local,
            "materialised_inside_window": materialised,
            "flush_vs_four_units": float((total - 4 * unit).abs().max()),
            "flush_vs_one_unit": float((total - unit).abs().max()),
            "magnitude": float(unit.abs().max())}


@case("trainer_fit_fsdp_accumulation")
def trainer_fit_fsdp_accumulation(rank: int, world: int, dist, options: dict) -> dict:
    """Drive the real ``Trainer.fit`` over FSDP and check what accumulation does.

    ``no_sync`` suppresses the per-microbatch gradient reduction, so replicas
    only agree if the reduction happens exactly once at the accumulation
    boundary.  The regression this guards is not in ``train_step`` but in
    ``fit``'s epilogue: the removed flush stepped on a trailing partial window
    whose microbatches were accumulated under ``no_sync`` and therefore never
    reduced, which diverged the replicas permanently.

    The batch count is deliberately not a multiple of ``grad_accumulation_steps``
    so a trailing partial window exists, and both ranks are fed identical data,
    so the reduced gradient equals the single-process gradient.  Running this
    same case at ``world_size == 1`` produces that reference.  It needs
    production FSDP with the real DP group -- the same path ``parallelize``
    builds -- rather than a hand-wrapped ``DistributedDataParallel``.
    """
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    torch.manual_seed(7)
    module = nn.Linear(4, 4, bias=True)
    values = {"grad_accumulation_steps": 3}
    if world > 1:
        values["parallel"] = {"dp_size": world}
        values["fsdp"] = {"enabled": True}
    config = FrameworkConfig.from_dict(values)
    runtime = Runtime(device="cpu", seed=7)
    model = parallelize(module, config=config, runtime=runtime) if world > 1 else module
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, config=config, loss_fn=_squared_error,
                      runtime=runtime)

    batches = [_batch(step, index) for step in range(7) for index in range(1)]
    trainer.fit(batches, epochs=1)
    applied = [step.optimizer_step for step in trainer.history]

    # FSDP shards the parameters, so each rank only holds its own slice of
    # ``module.weight``; reassemble the global tensor before comparing.  FSDP2
    # has no ``full_state_dict_context`` -- get_model_state_dict with
    # full_state_dict=True is the replacement, and it is collective.
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
    weight, bias = state["weight"], state["bias"]
    return {"rank": rank, "global_step": trainer.global_step, "applied": applied,
            "optimizer_steps": sum(applied),
            "weight": _plain(weight.detach()).flatten().tolist(),
            "bias": _plain(bias.detach()).flatten().tolist()}


@case("trainer_fit_fsdp_clipping")
def trainer_fit_fsdp_clipping(rank: int, world: int, dist, options: dict) -> dict:
    """Drive ``grad_clip_norm`` through the real ``Trainer`` over FSDP2.

    ``fsdp_grad_norm_is_global`` checks the ``clip_grad_norm_`` function.  This
    case checks the **wiring**: ``run_optimizer_step`` has to hand it the sharded
    module, at the point where DTensor gradients exist, and the reported norm has
    to match what one process reports for the same data.  A function that is
    right but called with the wrong object, or called before the accumulation
    window closes, passes the first check and fails this one.

    Both ranks are fed identical data, so the world_size=1 run of this same case
    is a real reference: FSDP2's globally-reduced gradient equals the
    single-process gradient, and the global norm -- hence the clip coefficient
    and every weight -- must match exactly.  Unlike the accumulation case, this
    one *does* discriminate: a shard-local norm is strictly smaller than the
    global one, so the coefficients would differ and the weights would diverge.

    The cap is deliberately small, and the reference's reported norms are
    asserted to exceed it, so the run cannot pass without clipping having
    actually taken place.
    """
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    cap = 0.05
    torch.manual_seed(7)
    module = nn.Linear(4, 4, bias=True)
    values = {"grad_accumulation_steps": 1, "grad_clip_norm": cap}
    if world > 1:
        values["parallel"] = {"dp_size": world}
        values["fsdp"] = {"enabled": True}
    config = FrameworkConfig.from_dict(values)
    runtime = Runtime(device="cpu", seed=7)
    model = parallelize(module, config=config, runtime=runtime) if world > 1 else module
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, config=config, loss_fn=_squared_error,
                      runtime=runtime)

    trainer.fit([_batch(step, 0) for step in range(5)], epochs=1)
    reported = [step.grad_norm for step in trainer.history]

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
    return {"rank": rank, "cap": cap, "grad_norms": reported,
            "optimizer_steps": trainer.optimizer_step,
            "weight": _plain(state["weight"].detach()).flatten().tolist(),
            "bias": _plain(state["bias"].detach()).flatten().tolist()}


def _squared_error(output, batch):
    from torch.nn import functional
    return functional.mse_loss(output, batch[1])


def _batch(step: int, index: int):
    import torch
    generator = torch.Generator().manual_seed(1000 * step + index)
    x = torch.randn(2, 4, generator=generator)
    y = torch.randn(2, 4, generator=generator)
    return x, y


# --------------------------------------------------------------------------- #
# Pipeline parallelism -- the production path, at pp_size > 1
# --------------------------------------------------------------------------- #
@case("pp_pipeline_step")
def pp_pipeline_step(rank: int, world: int, dist, options: dict) -> dict:
    """Drive the real ``Trainer.fit`` through the PP path and compare to world 1.

    Before this case existed, ``PipelineStage.pipeline_step`` and
    ``_pipeline_step_1f1b`` had **never been executed at pp_size > 1 by any test
    in the repository**: the only constructions of ``PipelineStage`` in tests
    pass ``pp_size=1``, which takes the plain-module shortcut above the schedule.
    The 17 in-process protocol tests cover ``parallel/pp_schedule.py``, which the
    production path does not call.  So the two implementations had disjoint
    coverage, and the one nothing covered was the one that runs.

    The model is a 3-child ``nn.Sequential`` so that a pp_size=2 split lands one
    Linear on each rank with the ReLU as the boundary.  Each rank reports its own
    stage's parameters; the test concatenates them and compares against the
    world_size=1 run of this same case, which reports the whole model.
    """
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    schedule = options.get("schedule", "gpipe")
    accumulation = int(options.get("accumulation", "1"))
    steps = int(options.get("steps", "3"))
    pp_size = 2 if world > 1 else 1

    torch.manual_seed(7)
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    values: dict[str, object] = {"grad_accumulation_steps": accumulation}
    if world > 1:
        values["parallel"] = {"pp_size": pp_size, "pp_schedule": schedule,
                              "num_microbatches": 2}
    config = FrameworkConfig.from_dict(values)
    runtime = Runtime(device="cpu", seed=7)
    model = parallelize(module, config=config, runtime=runtime) if world > 1 else module
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, config=config, runtime=runtime)

    trainer.fit([_classification_batch(step) for step in range(steps)], epochs=1)

    # Each rank holds only its own stage; world_size==1 holds everything.
    stage = model.module if hasattr(model, "module") else model
    parameters = _flatten(stage)
    applied = [step.optimizer_step for step in trainer.history]
    losses = None
    if rank == world - 1:                 # only the last stage computes a loss
        losses = [step.loss for step in trainer.history]
    return {"rank": rank, "global_step": trainer.global_step, "applied": applied,
            "optimizer_steps": sum(applied), "losses": losses,
            "parameter_count": len(parameters), "parameters": parameters}


def _classification_batch(step: int):
    """A (features, int-labels) tuple batch: the tuple branch of the loss policy."""
    import torch
    generator = torch.Generator().manual_seed(2000 + step)
    return torch.randn(4, 4, generator=generator), torch.randint(0, 4, (4,), generator=generator)


# --------------------------------------------------------------------------- #
# Checkpointing -- the persistence path, at world_size > 1
# --------------------------------------------------------------------------- #
def _named_projector():
    """A tiny model whose projections carry the names the TP plan matches on.

    The checkpoint cases used an anonymous ``nn.Sequential``, which the plan
    cannot name -- and ``parallelize`` now refuses an unnamed plan at ``tp_size>1``
    rather than sharding nothing.  So a TP-able model has to name its
    projections; that is a real requirement of the design, not a test detail.
    """
    from torch import nn

    class Projector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.activation = nn.ReLU()
            self.o_proj = nn.Linear(4, 4)

        def forward(self, value):
            return self.o_proj(self.activation(self.q_proj(value)))

    return Projector()


def _checkpoint_model(world: int, options: dict):
    """The model the checkpoint cases train, in whichever sharding they ask for."""
    import torch

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    torch.manual_seed(7)
    module = _named_projector()
    mode = options.get("mode", "fsdp")
    values: dict = {"grad_accumulation_steps": 1}
    if world > 1:
        if mode == "fsdp":
            values["parallel"] = {"dp_size": world}
            values["fsdp"] = {"enabled": True}
        elif mode == "tp":
            values["parallel"] = {"tp_size": world}
        else:
            raise ValueError(f"unknown checkpoint case mode {mode!r}")
    config = FrameworkConfig.from_dict(values)
    runtime = Runtime(device="cpu", seed=7)
    model = parallelize(module, config=config, runtime=runtime) if world > 1 else module
    return config, runtime, model


def _flatten(model) -> list:
    """Every parameter, flattened, as plain Python floats.

    FSDP2 holds parameters as DTensors, and ``.tolist()`` refuses tensor
    subclasses.  ``to_local()`` is the accessor that does not gather, which is
    what a per-rank before/after comparison wants: it checks that *this rank's*
    shard survived the round trip, and gathers nothing it would then have to
    ignore.
    """
    return [value for parameter in model.parameters()
            for value in _plain(parameter.detach()).flatten().tolist()]


def _plain(value):
    """A DTensor's local shard; a plain tensor passes through unchanged."""
    return value.to_local() if hasattr(value, "to_local") else value


@case("checkpoint_roundtrip")
def checkpoint_roundtrip(rank: int, world: int, dist, options: dict) -> dict:
    """Save through the production path, rebuild, load, and compare.

    The persistence path had **zero** multi-rank coverage: every checkpoint test
    was single-process, and nothing under ``tests/multirank/`` touched it.  That
    matters more than usual here because the design is per-rank: each rank writes
    its own complete ``rank_state.pt``, so correctness depends on every rank
    doing so and on ``load_state_dict`` behaving under a real process group.

    Each rank writes to its own path -- ``CheckpointManager.save`` issues no
    collective, so a shared path would have the ranks race over one directory.
    """
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer import Trainer

    config, runtime, model = _checkpoint_model(world, options)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # No loss_fn: `_classification_batch` carries int labels, which the shared
    # loss policy's tuple branch reduces with cross-entropy.
    trainer = Trainer(model, optimizer, config=config, runtime=runtime)
    trainer.fit([_classification_batch(0)], epochs=1)
    trained = _flatten(trainer.model.module)

    directory = Path(tempfile.mkdtemp(prefix=f"ckpt-{rank}-"))
    path = directory / "step"
    try:
        trainer.save_checkpoint(path)
        written = sorted(item.name for item in path.iterdir())

        # Rebuild from scratch and load; the fresh model must end up trained.
        _, runtime2, model2 = _checkpoint_model(world, options)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        trainer2 = Trainer(model2, optimizer2, config=config, runtime=runtime2)
        trainer2.load_checkpoint(path)
        restored = _flatten(trainer2.model.module)
        return {"rank": rank, "written": written, "trained": trained,
                "restored": restored, "global_step": trainer2.global_step,
                "optimizer_step": trainer2.optimizer_step,
                "max_diff": max(abs(a - b) for a, b in zip(trained, restored))
                            if len(trained) == len(restored) else float("inf")}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@case("checkpoint_dcp_roundtrip")
def checkpoint_dcp_roundtrip(rank: int, world: int, dist, options: dict) -> dict:
    """The genuinely sharded path, at world_size > 1.

    ``save_dcp``/``load_dcp`` wrap ``torch.distributed.checkpoint``, which really
    does split the state across ranks (each writes its own ``__N_M.distcp``
    shard).  Unlike the per-rank path above, DCP *is* collective, so every rank
    uses the SAME path -- that difference is the point of covering both.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer.checkpoint import CheckpointManager

    torch.manual_seed(11)
    original = torch.randn(8, 4)
    # ONE directory for the whole job -- `dcp.save` is collective and each rank
    # writes a different shard into it.  Keyed on MASTER_PORT so concurrent jobs
    # do not share one, and identical on every rank of this job.
    directory = Path(tempfile.gettempdir()) / f"aitrainer-dcp-{os.environ.get('MASTER_PORT', '0')}"
    path = directory / "step"
    if rank == 0:
        shutil.rmtree(directory, ignore_errors=True)
    dist.barrier()
    try:
        CheckpointManager().save_dcp(path, state={"w": original.clone()},
                                     metadata={"world": world})
        shards = sorted(name for name in os.listdir(path / "dcp") if name.endswith(".distcp"))

        restored = {"w": torch.zeros_like(original)}
        metadata = CheckpointManager().load_dcp(path, state=restored)
        return {"rank": rank, "shards": shards, "metadata": metadata,
                "max_diff": float((restored["w"] - original).abs().max().item()),
                "shape": list(restored["w"].shape)}
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(directory, ignore_errors=True)


@case("sharded_convert_and_load")
def sharded_convert_and_load(rank: int, world: int, dist, options: dict) -> dict:
    """Dense -> sharded manifest -> per-rank load, over a live process group.

    Covers the two entry points that make sharded loading possible and that only
    ever had single-process tests: ``CheckpointConverter.convert`` and
    ``ModelLoader.load_for_rank``.  Both are library-level; nothing in ``src/``
    calls either, and ``Trainer``'s own checkpoint path writes a full replica on
    every rank instead.  Verified here at pp=0/tp/dp coordinates so the claims
    about it are reproducible rather than one-off probes.

    ``partition_dims`` is the only thing that splits a tensor, and the split is
    always along the **TP** axis: the converter chunks by ``effective_tp`` and
    never by dp.  ``dp_size`` controls replica fan-out instead -- the same shard
    is written into every dp rank's file.  That is faithful to what DP means
    (replicas hold the same logical parameters), but it means a dp-only
    conversion saves no space, and ``convert(source, dest, world_size=N)`` with
    no explicit sizes means **tp=N**.  Asserting the recorded
    ``logical_sharding`` and the file contents is what keeps those two facts from
    being assumed rather than checked.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer.checkpoint import CheckpointConverter
    from aitrainer.checkpoint.reader import ModelLoader

    root = Path(tempfile.gettempdir()) / f"aitrainer-shard-{os.environ.get('MASTER_PORT', '0')}"
    source = root / "dense.pt"
    destination = root / "sharded"

    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(5)
        # `w` is named in partition_dims and gets split; `b` is not, and stays
        # full size in every rank's file -- the converter's actual contract.
        torch.save({"w": torch.randn(8, 4), "b": torch.randn(8)}, source)
    dist.barrier()

    if rank == 0:
        CheckpointConverter(partition_dims={"w": 0}).convert(
            source, destination, world_size=world, tp_size=world)
    dist.barrier()

    reference = torch.load(source, map_location="cpu", weights_only=False)
    local = {"w": torch.zeros(8 // world, 4), "b": torch.zeros(8)}
    stats = ModelLoader().load_for_rank(destination / "manifest.json", local,
                                        world_size=world, tp_rank=rank)

    half = 8 // world
    expected = reference["w"][rank * half:(rank + 1) * half]
    _require(torch.equal(local["w"], expected),
             f"rank {rank} did not receive ref[{rank * half}:{(rank + 1) * half}]")
    _require(torch.equal(local["b"], reference["b"]),
             "an unpartitioned tensor must arrive full-size on every rank")

    manifest = ModelLoader().read_manifest(destination / "manifest.json")
    shard_files = sorted({Path(shard.source_file).name
                          for shards in manifest.tensors.values() for shard in shards})
    return {"rank": rank, "stats": stats,
            "logical_sharding": dict(manifest.logical_sharding),
            "shard_files": shard_files,
            "w_shape": list(local["w"].shape), "b_shape": list(local["b"].shape)}


@case("sharded_convert_replicates_along_dp")
def sharded_convert_replicates_along_dp(rank: int, world: int, dist, options: dict) -> dict:
    """``dp_size`` fans out copies; it does not shard.

    Worth pinning because the name suggests otherwise.  With ``tp_size=1`` and
    ``dp_size=2`` every rank's file holds the WHOLE tensor, so a dp-only
    conversion gives no storage saving -- which follows from what DP means, but
    is not what a reader of ``--dp-size`` would guess.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer.checkpoint import CheckpointConverter
    from aitrainer.checkpoint.reader import ModelLoader

    root = Path(tempfile.gettempdir()) / f"aitrainer-dp-{os.environ.get('MASTER_PORT', '0')}"
    source = root / "dense.pt"
    destination = root / "converted"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(5)
        torch.save({"w": torch.randn(8, 4)}, source)
    dist.barrier()
    if rank == 0:
        CheckpointConverter(partition_dims={"w": 0}).convert(
            source, destination, world_size=world, dp_size=world, tp_size=1)
    dist.barrier()

    reference = torch.load(source, map_location="cpu", weights_only=False)
    local = {"w": torch.zeros(8, 4)}
    ModelLoader().load_for_rank(destination / "manifest.json", local,
                                world_size=world, dp_rank=rank)
    return {"rank": rank, "shape": list(local["w"].shape),
            "matches_whole_tensor": bool(torch.equal(local["w"], reference["w"]))}


@case("sharded_load_rejects_a_foreign_coordinate")
def sharded_load_rejects_a_foreign_coordinate(rank: int, world: int, dist, options: dict) -> dict:
    """Asking for coordinates the manifest never addressed must fail loudly.

    The dangerous alternative is silent: the model keeps its random
    initialisation while the load reports success.  ``load_for_rank`` raises
    ``KeyError`` for that case -- distinct from a tensor this stage simply does
    not own, which is skipped -- and this pins the difference.

    ``(pp, tp, dp)`` is positional, so ``dp_rank=N`` on a tp-sharded manifest
    asks for a tuple nothing targets.  Note the converse trap this test would
    fall into if written carelessly: on rank 0, ``dp_rank=0`` is the SAME tuple
    as ``tp_rank=0``, so it legitimately matches.  The foreign coordinate has to
    be one no spec uses.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer.checkpoint import CheckpointConverter
    from aitrainer.checkpoint.reader import ModelLoader

    root = Path(tempfile.gettempdir()) / f"aitrainer-foreign-{os.environ.get('MASTER_PORT', '0')}"
    source = root / "dense.pt"
    destination = root / "sharded"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(5)
        torch.save({"w": torch.randn(8, 4)}, source)
    dist.barrier()
    if rank == 0:
        CheckpointConverter(partition_dims={"w": 0}).convert(source, destination, world_size=world)
    dist.barrier()

    manifest = ModelLoader().read_manifest(destination / "manifest.json")
    targets = sorted({shard.target_rank for shards in manifest.tensors.values() for shard in shards})
    foreign = (0, 0, 1)                 # manifest addresses tp ranks; dp=1 targets nothing
    _require(foreign not in targets,
             f"the chosen coordinate {foreign} is a real target in {targets}; "
             "this case would prove nothing")
    try:
        ModelLoader().load_for_rank(destination / "manifest.json",
                                    {"w": torch.zeros(8 // world, 4)},
                                    world_size=world, dp_rank=foreign[2])
    except KeyError as exc:
        return {"rank": rank, "refused": True, "targets": targets,
                "message": str(exc)[:160]}
    raise AssertionError(
        f"rank {rank}: foreign coordinate {foreign} loaded without complaint; the "
        "model would have kept its random initialisation")


# --------------------------------------------------------------------------- #
def _tp_group(world: int, rank: int):
    from aitrainer.parallel.groups import ProcessGroups
    from aitrainer.topology import RankMapping
    mapping = RankMapping(world, pp_size=1, dp_size=1, tp_size=world,
                          global_ranks=tuple(range(world)))
    return ProcessGroups.create(mapping).tp_group


@case("data_parallel_group_is_handed_over")
def data_parallel_group_is_handed_over(rank: int, world: int, dist, options: dict) -> dict:
    """``Trainer.dp_process_group()`` must hand the data provider a usable group.

    ``loop.fit`` called ``data.train_dataloader(dp_group=None, ...)``.  A provider
    reading that argument sees no DP axis, so it cannot tell the ranks apart and
    every rank walks the whole dataset in the same order: data parallelism
    degenerates into N replicas averaging identical gradients.  It still trains
    and the loss still falls, so nothing reports that N times the compute buys
    one rank's batch.

    A non-None group is not enough on its own.  What a provider shards with is
    ``dist.get_rank(group)`` -- that index has to differ per rank, or every rank
    still reads shard 0 of a group that happens to exist.  Both are reported.
    """
    import torch

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    # Data parallelism is only valid with a gradient-reducing model, so the two
    # are configured together -- this is the shape a real run has.
    config = FrameworkConfig.from_dict({"parallel": {"dp_size": world}, "fsdp": {"enabled": True}})
    torch.manual_seed(11)
    module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    runtime = Runtime(device="cpu", seed=11)
    wrapped = parallelize(module, config=config, runtime=runtime)
    trainer = Trainer(wrapped, torch.optim.SGD(wrapped.parameters(), lr=0.01),
                      config=config, runtime=runtime)
    group = trainer.dp_process_group()
    if group is None:
        raise AssertionError(
            "dp_size>1 but the data provider was handed dp_group=None, so no rank "
            "can shard the dataset")
    index = dist.get_rank(group)

    # The group existing is only half of it: ``loop.fit`` and
    # ``evaluation.evaluate`` are the call sites that were passing ``None``, so
    # ask them what the provider actually receives.  A recording provider is the
    # only way to see that, and it is the difference between "the helper works"
    # and "the helper is used".
    handed: dict = {}

    class RecordingProvider:
        def train_dataloader(self, *, dp_group=None, seed=None):
            handed["dp_group"] = dp_group
            handed["seed"] = seed
            return [(torch.randn(4, 8, generator=torch.Generator().manual_seed(1)),
                     torch.randint(0, 4, (4,), generator=torch.Generator().manual_seed(1)))]

    trainer.fit(RecordingProvider(), epochs=1)
    fit_group = handed.get("dp_group")
    handed.clear()
    trainer.evaluate(RecordingProvider())
    eval_group = handed.get("dp_group")

    gathered: list = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "index": index,
                                      "fit": fit_group is group,
                                      "evaluate": eval_group is group})
    return {"rank": rank, "group": str(group),
            "group_size": dist.get_world_size(group), "index": index, "seed": 11,
            "fit_handed_it": fit_group is group,
            "evaluate_handed_it": eval_group is group,
            "indices": {str(item["rank"]): item["index"] for item in gathered}}


@case("sharded_checkpoint_roundtrip")
def sharded_checkpoint_roundtrip(rank: int, world: int, dist, options: dict) -> dict:
    """``Trainer.save_sharded`` / ``load_sharded`` at world_size > 1.

    The opt-in DCP path, exercised end to end: train, save collectively, rebuild
    from scratch, load, and compare parameters AND bookkeeping.  AdamW is used
    rather than SGD so the optimizer genuinely has state to carry -- with plain
    SGD ``get_optimizer_state_dict`` returns an empty state and the test would
    pass while proving nothing about optimizer persistence.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    path = Path(tempfile.gettempdir()) / f"aitrainer-sharded-{os.environ.get('MASTER_PORT', '0')}"
    config = FrameworkConfig.from_dict({"parallel": {"dp_size": world}, "fsdp": {"enabled": True}})

    def build():
        # A FRESH module each time: `parallelize` FSDP-wraps in place, so reusing
        # one object would wrap an already-wrapped model and the state-dict
        # shapes would differ between save and load.
        torch.manual_seed(7)
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
        runtime = Runtime(device="cpu", seed=7)
        wrapped = parallelize(module, config=config, runtime=runtime)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.01)
        trainer = Trainer(wrapped, optimizer, config=config, runtime=runtime)
        return trainer

    trainer = build()
    batches = []
    for index in range(3):
        generator = torch.Generator().manual_seed(3000 + index)
        batches.append((torch.randn(4, 8, generator=generator),
                        torch.randint(0, 4, (4,), generator=generator)))
    trainer.fit(batches, epochs=1)
    before = _flatten(trainer.model.module)

    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    trainer.save_sharded(path)
    dist.barrier()
    written = sorted(item.name for item in path.iterdir())
    shards = sorted(name for name in os.listdir(path / "dcp") if name.endswith(".distcp"))

    trainer2 = build()
    restored = trainer2.load_sharded(path)
    after = _flatten(trainer2.model.module)
    dist.barrier()
    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)
    # Measure the optimizer state itself, not the metadata blob it travels with.
    # Zero entries would mean the checkpoint is not resumable.
    optimizer_state_entries = sum(len(entry) for entry in trainer2.optimizer.state.values())
    return {"rank": rank, "written": written, "shards": shards,
            "global_step": restored["global_step"], "optimizer_step": restored["optimizer_step"],
            "format": (restored["metadata"] or {}).get("format"),
            "optimizer_state_entries": optimizer_state_entries,
            "max_diff": max(abs(a - b) for a, b in zip(before, after))
                        if len(before) == len(after) else float("inf")}


@case("load_pretrained_per_rank")
def load_pretrained_per_rank(rank: int, world: int, dist, options: dict) -> dict:
    """Load pretrained weights through the parallelised model, each rank reading its own.

    The thing being checked is not just that the weights arrive -- it is *how*.
    Every rank reads only the shards addressed to its own ``(pp, tp, dp)``
    coordinate and writes them straight into the parameters it already holds, so
    no rank ever reads a complete model and nothing is broadcast.  One rank
    reading everything and shipping it out is the alternative this exists to
    avoid, and it is invisible in the final weights -- which is why the worker
    reports ``bytes_read`` per rank.

    ``options`` carries ``dp``/``tp``/``pp``; the model is the four-unit block
    used elsewhere so a split up to pp=4 is possible.
    """
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    from torch import nn

    from aitrainer import (
        CheckpointConverter,
        FrameworkConfig,
        Runtime,
        Trainer,
        TransformerTPPlan,
        stage_assignment,
    )
    from aitrainer.parallelizer import parallelize

    dp = int(options.get("dp", 1))
    tp = int(options.get("tp", 1))
    pp = int(options.get("pp", 1))

    class Unit(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden)
            self.o_proj = nn.Linear(hidden, hidden)

        def forward(self, value):
            return self.o_proj(torch.relu(self.q_proj(value)))

    class Block(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.unit0 = Unit(hidden)
            self.unit1 = Unit(hidden)
            self.unit2 = Unit(hidden)
            self.unit3 = Unit(hidden)

        def forward(self, value):
            for unit in (self.unit0, self.unit1, self.unit2, self.unit3):
                value = unit(value)
            return value

    path = Path(tempfile.gettempdir()) / f"aitrainer-pretrained-{os.environ.get('MASTER_PORT', '0')}"
    # The "pretrained" weights every rank can reproduce independently, standing in
    # for a dense checkpoint on shared storage.
    torch.manual_seed(31337)
    reference = Block()
    dense = {name: value.detach().clone() for name, value in reference.state_dict().items()}

    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        torch.save(reference.state_dict(), path / "dense.pt")
        plan = TransformerTPPlan()
        CheckpointConverter(partition_dims=plan.partition_dims(reference) if tp > 1 else None).convert(
            path / "dense.pt", path / "sharded", pp_size=pp, dp_size=dp, tp_size=tp,
            dp_sharded=dp > 1, stage_assignment=stage_assignment(reference, pp) if pp > 1 else None)
    dist.barrier()

    # A FRESH model: what it initialises to must be irrelevant once loaded.
    torch.manual_seed(5)
    config = FrameworkConfig.from_dict({
        "parallel": {"dp_size": dp, "tp_size": tp, "pp_size": pp,
                     "pp_schedule": "gpipe" if pp > 1 else "none",
                     "num_microbatches": 2 if pp > 1 else 1},
        "fsdp": {"enabled": dp > 1},
    })
    runtime = Runtime(device="cpu", init_process_group=False, seed=5)
    model = parallelize(Block(), config=config, runtime=runtime)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                      config=config, runtime=runtime,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    stats = trainer.load_pretrained(path / "sharded" / "manifest.json")

    inner = trainer.model.module
    module = inner.module if hasattr(inner, "stage_id") else inner
    worst = 0.0
    for name, parameter in module.named_parameters():
        got = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        worst = max(worst, float((got.detach() - dense[name]).abs().max()))
    gathered = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "bytes": stats["bytes_read"],
                                      "tensors": stats["tensors_loaded"], "stage": getattr(inner, "stage_id", 0)})
    dist.barrier()
    return {"rank": rank, "stage": getattr(inner, "stage_id", 0),
            "max_diff": worst, "bytes_read": stats["bytes_read"],
            "tensors_loaded": stats["tensors_loaded"],
            "all_ranks": gathered if rank == 0 else None,
            "dense_bytes": sum(v.numel() * v.element_size() for v in dense.values())}


@case("composition_train_matches_dense")
def composition_train_matches_dense(rank: int, world: int, dist, options: dict) -> dict:
    """Train through ``Trainer.fit`` on every axis at once, against a dense reference.

    The checkpoint cases prove a round trip is lossless; this proves the
    *training* on the compounded axes is numerically right.  Every rank gets the
    same data, so a DP reduction over identical replicas is the identity on the
    gradient and the single-process run of this same case is a valid reference:
    TP/SP must reproduce it exactly, and the pipeline stages' parameters,
    concatenated in stage order, must equal the whole model's.

    ``options["shape"]`` is ``dp,tp,pp`` (``sp`` optional), so the same case
    covers ``tp=2,pp=2`` and, at world 8, ``dp=2,tp=2,pp=2`` with and without SP.
    """
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    parts = options.get("shape", "1,1,1").split(",")
    dp, tp, pp = (int(part) for part in parts)
    sp = options.get("sp", "none")

    class Unit(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.q_proj = nn.Linear(hidden, hidden)
            self.o_proj = nn.Linear(hidden, hidden)

        def forward(self, value):
            return self.o_proj(torch.relu(self.q_proj(self.norm(value))))

    class Block(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.unit0 = Unit(hidden)
            self.unit1 = Unit(hidden)
            self.unit2 = Unit(hidden)
            self.unit3 = Unit(hidden)

        def forward(self, value):
            for unit in (self.unit0, self.unit1, self.unit2, self.unit3):
                value = unit(value)
            return value

    values = {"parallel": {"dp_size": dp, "tp_size": tp, "pp_size": pp, "sp_backend": sp},
              "grad_accumulation_steps": 1,
              "fsdp": {"enabled": dp > 1}}
    if pp > 1:
        values["parallel"]["pp_schedule"] = "gpipe"
        values["parallel"]["num_microbatches"] = 2
    config = FrameworkConfig.from_dict(values)
    torch.manual_seed(1234)
    probe_before = float(torch.randn(1).item())
    runtime = Runtime(device="cpu", init_process_group=False, seed=1234)
    probe_after = float(torch.randn(1).item())
    block = Block()
    before_parallelize = [value for param in block.parameters()
                          for value in param.detach().flatten().tolist()]
    model = parallelize(block, config=config, runtime=runtime) if world > 1 else block
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, config=config, runtime=runtime,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    generator = torch.Generator().manual_seed(88)
    data = [(torch.randn(2, 4, 8, generator=generator),
             torch.randn(2, 4, 8, generator=generator)) for _ in range(2)]
    def snapshot() -> list:
        return [value for p in trainer.model.module.parameters()
                for value in _full(p.detach()).flatten().tolist()]

    initial = snapshot()
    trainer.fit(data, epochs=1)
    return {"rank": rank, "shape": options.get("shape"), "sp": sp,
            "stage": getattr(trainer.model.module, "stage_id", 0),
            "initial": initial, "parameters": snapshot(),
            "names": [name for name, _ in (
                trainer.model.module.module
                if hasattr(trainer.model.module, "stage_id") else trainer.model.module
            ).named_parameters()],
            "probe_before": probe_before, "probe_after": probe_after,
            "before_parallelize": before_parallelize,
            "initial_seed": torch.initial_seed(),
            "optimizer_steps": trainer.optimizer_step}


@case("sharded_roundtrip_axes")
def sharded_roundtrip_axes(rank: int, world: int, dist, options: dict) -> dict:
    """Round-trip a sharded checkpoint over any combination of the four axes.

    ``options`` names ``dp`` / ``tp`` / ``pp`` / ``sp``; the model has four
    top-level children so a split up to pp=4 is possible, and its projections
    carry the names the TP plan matches on.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    dp = int(options.get("dp", 1))
    tp = int(options.get("tp", 1))
    pp = int(options.get("pp", 1))
    sp = options.get("sp", "none")

    class Unit(nn.Module):
        """One norm + col->row pair: the smallest self-contained TP unit.

        The ``nn.LayerNorm`` is load-bearing for the SP tests.  Without a norm
        ``sequence_parallel_styles`` returns an empty dict, so ``sp_backend`` had
        nothing to act on and a "tp+sp" case would have been passing vacuously
        with SP silently off.
        """

        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.q_proj = nn.Linear(hidden, hidden)   # column
            self.o_proj = nn.Linear(hidden, hidden)   # row

        def forward(self, value):
            return self.o_proj(torch.relu(self.q_proj(self.norm(value))))

    class Block(nn.Module):
        """Four top-level children, each a complete col->row pair.

        The pairing has to stay *inside* a stage.  A column projection emits a
        sharded activation, and only its paired row projection can consume that
        -- across a pipeline boundary the activation is a full tensor, so a row
        projection on the far side would treat it as a shard.  Measured: putting
        the column and its row in different stages deadlocks inside ``fit``
        (Gloo recv timeout at the barrier after it), rather than failing a
        shape check.  Nesting one pair per child keeps every split point valid
        for pp up to 4.
        """

        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.unit0 = Unit(hidden)
            self.unit1 = Unit(hidden)
            self.unit2 = Unit(hidden)
            self.unit3 = Unit(hidden)

        def forward(self, value):
            for unit in (self.unit0, self.unit1, self.unit2, self.unit3):
                value = unit(value)
            return value

    values = {"parallel": {"dp_size": dp, "tp_size": tp, "pp_size": pp, "sp_backend": sp},
              "fsdp": {"enabled": dp > 1}}
    if pp > 1:
        values["parallel"]["pp_schedule"] = "gpipe"
        values["parallel"]["num_microbatches"] = 2
    config = FrameworkConfig.from_dict(values)
    path = Path(tempfile.gettempdir()) / (
        f"aitrainer-axes-{dp}-{tp}-{pp}-{sp}-{os.environ.get('MASTER_PORT', '0')}")

    def build():
        torch.manual_seed(7)
        runtime = Runtime(device="cpu", seed=7)
        model = parallelize(Block(), config=config, runtime=runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, config=config, runtime=runtime,
                          loss_fn=lambda output, batch: torch.nn.functional.mse_loss(
                              output, batch[1]))
        return trainer

    # Everything from the very first build is inside the try: a combination the
    # framework refuses (pp+tp) fails during `parallelize`, and the case has to
    # report that as an answer rather than as a crashed worker.
    try:
        trainer = build()
        generator = torch.Generator().manual_seed(53)
        trainer.fit([(torch.randn(2, 8, generator=generator),
                      torch.randn(2, 8, generator=generator)) for _ in range(2)], epochs=1)
        trained = _flatten(trainer.model.module)
        inner = trainer.model.module
        keys = sorted(name for name, _ in (inner.module if hasattr(inner, "stage_id")
                                           else inner).named_parameters())
        gathered: list = [None] * world
        dist.all_gather_object(gathered, {"trained": trained, "keys": keys})
        per_rank_values_differ = any(gathered[0]["trained"] != other["trained"]
                                     for other in gathered[1:])
        keys_collide = all(gathered[0]["keys"] == other["keys"] for other in gathered[1:])

        if rank == 0:
            shutil.rmtree(path, ignore_errors=True)
        dist.barrier()
        trainer.save_sharded(path)
        dist.barrier()
        trainer2 = build()
        trainer2.load_sharded(path)
        restored = _flatten(trainer2.model.module)
        return {"rank": rank, "ran": True, "dp": dp, "tp": tp, "pp": pp, "sp": sp,
                "max_diff": max(abs(a - b) for a, b in zip(trained, restored)),
                "steps": trainer2.global_step,
                "keys": keys, "keys_collide_across_ranks": keys_collide,
                "per_rank_values_differ": per_rank_values_differ,
                "optimizer_state_entries": len(list(trainer2.optimizer.state.values()))}
    except (ValueError, RuntimeError) as exc:
        # ValueError covers the framework's own refusals (TPConfigurationError,
        # ConfigurationError); RuntimeError covers torch's distributed errors.
        # Narrowing it keeps a genuine bug from being reported as a supported
        # answer -- an unexpected exception should crash the worker, not be
        # collected as "this combination is refused".
        return {"rank": rank, "ran": False, "dp": dp, "tp": tp, "pp": pp, "sp": sp,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


@case("sharded_reshard_to_one_process")
def sharded_reshard_to_one_process(rank: int, world: int, dist, options: dict) -> dict:
    """The sharded checkpoint must be readable at a different world size.

    This is what "真分片" buys and what the framework did not have: the old
    refusal was documented as covering "换 world size 只能拒绝，不能转换".  Because
    the saved tensors are globally described, DCP can hand the whole thing to a
    single process with no process group at all (``no_dist=True``), which is a
    simultaneous change of world size AND of TP layout.

    ``get_model_state_dict(full_state_dict=True)`` is collective, so every rank
    calls it -- calling it on rank 0 alone deadlocks the others at the next
    barrier, which is exactly how the first version of this probe hung.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    from torch import nn
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.activation = nn.ReLU()
            self.o_proj = nn.Linear(8, 8)

        def forward(self, value):
            return self.o_proj(self.activation(self.q_proj(value)))

    config = FrameworkConfig.from_dict({"parallel": {"tp_size": world}})
    path = Path(tempfile.gettempdir()) / f"aitrainer-reshard-{os.environ.get('MASTER_PORT', '0')}"
    torch.manual_seed(7)
    runtime = Runtime(device="cpu", seed=7)
    model = parallelize(Block(), config=config, runtime=runtime)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=0.01),
                      config=config, runtime=runtime,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    generator = torch.Generator().manual_seed(41)
    trainer.fit([(torch.randn(2, 8, generator=generator),
                  torch.randn(2, 8, generator=generator)) for _ in range(2)], epochs=1)

    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    trainer.save_sharded(path)
    dist.barrier()

    # Collective: every rank must join, even though only rank 0 uses the result.
    trained = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
    result = {"rank": rank, "world": world, "loaded": False, "max_diff": None,
              "shape": None, "keys": sorted(trained)}
    if rank == 0:
        import torch.distributed.checkpoint as dcp
        dense = Block().state_dict()
        dcp.load({"model": dense}, storage_reader=dcp.FileSystemReader(str(path / "dcp")),
                 no_dist=True)
        worst = max(float((dense[name].cpu() - trained[name].cpu()).abs().max())
                    for name in trained)
        result.update({"loaded": True, "max_diff": worst,
                       "shape": list(dense["q_proj.weight"].shape),
                       "shapes_match": all(tuple(dense[n].shape) == tuple(trained[n].shape)
                                           for n in trained)})
    dist.barrier()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--shape", default="2,2,1",
                        help="pp,dp,tp for the process-group cases")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="case-specific option; repeatable")
    arguments = parser.parse_args(argv)

    import torch  # noqa: F401 - imported so a missing torch fails loudly here
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=arguments.timeout_seconds))
    try:
        options = {"shape": arguments.shape}
        for item in arguments.option:
            key, _, value = item.partition("=")
            options[key] = value
        result = CASES[arguments.case](rank, world, dist, options)
        dist.barrier()
        print("RESULT " + json.dumps(result), flush=True)
    except Exception as exc:  # noqa: BLE001 - reported to the parent process
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
