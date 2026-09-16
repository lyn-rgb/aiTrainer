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
# §4.10 b -- ColumnParallelLinear gather_output + bias
# --------------------------------------------------------------------------- #
@case("column_parallel_bias")
def column_parallel_bias(rank: int, world: int, dist, options: dict) -> dict:
    """A gathered column-parallel layer must equal its dense reference.

    The bug it guards: ``self.bias`` holds ``output_size_per_partition`` entries,
    so adding it AFTER ``gather_output`` mismatched the full width by tp_size.
    """
    import torch
    from torch import nn

    from aitrainer.parallel.tp import ColumnParallelLinear

    group = _tp_group(world, rank)
    torch.manual_seed(20240915)
    dense = nn.Linear(8, 12, bias=True)

    gathered = ColumnParallelLinear.from_dense(dense, process_group=group, gather_output=True)
    x = torch.randn(4, 8)
    out = gathered(x)
    _require(tuple(out.shape) == (4, 12), f"gathered output shape {tuple(out.shape)} != (4, 12)")
    error = _max_abs_diff(out, dense(x))

    fused = ColumnParallelLinear.from_dense(dense, process_group=group, gather_output=True,
                                            skip_bias_add=True)
    fused_out, fused_bias = fused(x)
    _require(tuple(fused_bias.shape) == (12,),
             f"skip_bias_add bias width {tuple(fused_bias.shape)} != (12,)")
    fused_error = _max_abs_diff(fused_out + fused_bias, dense(x))

    # Input gradients must match the dense layer's, which is what makes the
    # gather's backward pass (an all-gather in reverse) correct.
    x_parallel, x_dense = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    gathered(x_parallel).sum().backward()
    dense(x_dense).sum().backward()
    grad_error = _max_abs_diff(x_parallel.grad, x_dense.grad)
    return {"rank": rank, "forward_error": error, "fused_error": fused_error,
            "input_grad_error": grad_error}


# --------------------------------------------------------------------------- #
# §4.10 c -- SequenceParallelLayerNorm parameter-gradient reduction
# --------------------------------------------------------------------------- #
@case("sp_layernorm_grad_reduction")
def sp_layernorm_grad_reduction(rank: int, world: int, dist, options: dict) -> dict:
    """Replicated norm parameters must receive the TP-reduced gradient.

    Each rank only sees its own token shard, so without a reduction the weight
    and bias copies diverge after the first optimizer step even though the
    forward output and the input gradient are already correct.
    """
    import torch
    from torch import nn

    from aitrainer.parallel.sp_sequence import SequenceParallelLayerNorm

    group = _tp_group(world, rank)
    hidden, batch, length = 6, 2, 8
    torch.manual_seed(4242)

    dense_norm = nn.LayerNorm(hidden)
    value = torch.randn(batch, length, hidden)

    parallel = SequenceParallelLayerNorm(hidden, process_group=group, input_is_parallel=True)
    with torch.no_grad():
        parallel.norm.weight.copy_(dense_norm.weight)
        parallel.norm.bias.copy_(dense_norm.bias)

    chunk = length // world
    local_value = value[:, rank * chunk:(rank + 1) * chunk, :].clone().requires_grad_(True)
    local_out = parallel(local_value)

    dense_value = value.clone().requires_grad_(True)
    dense_out = dense_norm(dense_value)
    reference_shard = dense_out[:, rank * chunk:(rank + 1) * chunk, :]
    forward_error = _max_abs_diff(local_out, reference_shard)

    local_out.sum().backward()
    dense_out.sum().backward()
    weight_error = _max_abs_diff(parallel.norm.weight.grad, dense_norm.weight.grad)
    bias_error = _max_abs_diff(parallel.norm.bias.grad, dense_norm.bias.grad)

    # The same reduction must hold for the input gradient, which is the other
    # half of the contract.
    input_grad_ok = _max_abs_diff(local_value.grad,
                                  dense_value.grad[:, rank * chunk:(rank + 1) * chunk, :]) < 1e-6
    _require(input_grad_ok, "local input gradient does not match the dense reference shard")
    return {"rank": rank, "forward_error": forward_error, "weight_grad_error": weight_error,
            "bias_grad_error": bias_error}


@case("sp_layernorm_without_reduction_diverges")
def sp_layernorm_without_reduction_diverges(rank: int, world: int, dist, options: dict) -> dict:
    """Negative control for the reduction: an unreduced replica must diverge.

    Runs the identical sharded computation through a plain ``nn.LayerNorm``, so
    an unreduced rank-local gradient is what it produces.  The harness asserts
    the two ranks disagree, which is what makes the reduction load-bearing.
    """
    import torch
    from torch import nn

    hidden, batch, length = 6, 2, 8
    torch.manual_seed(4242)
    norm = nn.LayerNorm(hidden)
    value = torch.randn(batch, length, hidden)
    chunk = length // world
    local = value[:, rank * chunk:(rank + 1) * chunk, :].detach().requires_grad_(True)
    norm(local).sum().backward()
    return {"rank": rank, "weight_grad": norm.weight.grad.tolist()}


# --------------------------------------------------------------------------- #
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
    """``wrap_fsdp`` must be able to initialise FSDP with CPU parameters.

    The bug it guards: ``wrap_fsdp`` forwarded ``device_id`` to FSDP only when it
    started with "cuda".  With no device_id, FSDP infers one, and on a CPU-only
    macOS host that inference resolves the custom 'mps' backend and raises
    ``AttributeError: Custom backend 'mps' not implement
    torch.mps.current_device`` -- so the STABLE ``fsdp_full_shard`` capability
    could not be initialised at all, with an error naming neither FSDP nor the
    missing argument.
    """
    import torch

    from aitrainer.parallel.fsdp import wrap_fsdp

    class _Runtime:
        world_size = world

        class state:
            device = "cpu"

    wrapped = wrap_fsdp(torch.nn.Linear(4, 4), runtime=_Runtime(), mesh=None, config=None)
    wrapped(torch.randn(2, 4)).sum().backward()
    return {"rank": rank, "wrapped": type(wrapped).__name__}


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
    # ``module.weight``; gather the full tensor before comparing.
    if world > 1:
        with model.full_state_dict_context():
            state = model.state_dict()
        weight, bias = state["weight"], state["bias"]
    else:
        weight, bias = module.weight.detach(), module.bias.detach()
    return {"rank": rank, "global_step": trainer.global_step, "applied": applied,
            "optimizer_steps": sum(applied),
            "weight": weight.detach().flatten().tolist(),
            "bias": bias.detach().flatten().tolist()}


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
    parameters = [value for parameter in stage.parameters()
                  for value in parameter.detach().flatten().tolist()]
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
def _checkpoint_model(world: int, options: dict):
    """The model the checkpoint cases train, in whichever sharding they ask for."""
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime
    from aitrainer.parallelizer import parallelize

    torch.manual_seed(7)
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
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
    return [value for parameter in model.parameters()
            for value in parameter.detach().flatten().tolist()]


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


# --------------------------------------------------------------------------- #
def _tp_group(world: int, rank: int):
    from aitrainer.parallel.groups import ProcessGroups
    from aitrainer.topology import RankMapping
    mapping = RankMapping(world, pp_size=1, dp_size=1, tp_size=world,
                          global_ranks=tuple(range(world)))
    return ProcessGroups.create(mapping).tp_group


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
