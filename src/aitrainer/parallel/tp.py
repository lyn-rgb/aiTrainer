"""Tensor parallelism through torch's DTensor parallel styles.

This used to be a hand-written pair of layers (``ColumnParallelLinear`` /
``RowParallelLinear``) that did their own collectives through custom
``autograd.Function`` wrappers.  Their parameters were ordinary
``nn.Parameter`` holding a *shard* -- and nothing anywhere recorded that the
shard was a slice of a larger logical tensor.  That is the defect this module's
rewrite exists to remove: ``get_model_state_dict`` returned each rank's local
tensor, ``torch.distributed.checkpoint`` then treated same-named keys as one
logical tensor, and the last rank to write won.

With ``parallelize_module`` the parameter *is* a ``DTensor``.  The module tree
is not restructured -- ``q_proj`` stays the ``nn.Linear`` it always was, with
its weight replaced by a DTensor -- so state-dict keys are unchanged and the
global shape is recoverable.  Measured at tp=2: ``q_proj.weight`` is a DTensor
of global shape (8, 8) sharded to (4, 8) per rank, and
``get_model_state_dict(..., StateDictOptions(full_state_dict=True))`` returns a
plain (8, 8) tensor equal to the dense model's.

Two consequences shape the code below:

* **An empty plan is not a no-op.**  ``parallelize_module(module, mesh, {})``
  succeeds and shards nothing, so ``tp_size>1`` would train N identical full
  replicas -- no communication, no error, just wrong.  :func:`parallelize_tensor_parallel`
  refuses that.
* **The styles come from the caller.**  Which projection is column-wise is a
  statement about the model, so it lives in ``plugins/transformer.py``; this
  module only knows how to apply it.
"""

from __future__ import annotations

from typing import Any


class TPConfigurationError(ValueError):
    """Raised when a TP plan cannot be applied to the module it names."""


def device_mesh(mesh: Any, process_group: Any = None, device_type: str = "cpu") -> Any:
    """The 1-D TP DeviceMesh, however the caller described it.

    ``parallelize_module`` takes a DeviceMesh, never a ProcessGroup.
    ``DeviceMeshManager`` exposes a real DeviceMesh for the ``(pp, dp, tp)``
    topology only when the global ranks are in logical order; for a permuted
    mapping it deliberately keeps process groups instead (``mesh.py``), so
    ``from_group`` covers that case.  Same split as ``parallel.fsdp``.
    """
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    if isinstance(process_group, DeviceMesh):
        return process_group
    if process_group is not None:
        return DeviceMesh.from_group(process_group, device_type)
    if isinstance(mesh, DeviceMesh):
        return mesh
    import torch.distributed as dist
    if not dist.is_initialized():
        raise TPConfigurationError(
            "tensor parallelism needs an initialized process group; torch cannot "
            "build a device mesh without one. Initialize a process group first -- "
            "a single-rank one is enough -- then parallelize the module.")
    return DeviceMesh(device_type, torch.arange(dist.get_world_size(), dtype=torch.int))


def _sharded_norm_style(sequence_dim: int) -> Any:
    """A style that shards a **replicated** input, runs the module, and gathers back.

    This is not ``torch.distributed.tensor.parallel.SequenceParallel``, and the
    difference is the whole reason this function exists.  ``SequenceParallel``
    *assumes* the input is already a per-rank slice of the sequence and merely
    annotates it (``DTensor.from_local(..., run_check=False)``).  Feeding it a
    replicated tensor therefore does not shard anything: the tensor comes back
    labelled ``Shard(1)`` with a global sequence dimension that is
    ``world_size`` times too large, and nothing raises.  Measured at world=2
    with a (2, 4, 8) input: the norm returned a DTensor claiming global shape
    (2, 8, 8).  The failure surfaced one module later, as

        Sharding propagation failed for Op(op=aten.view.default,
        args_schema=Spec(S(1) on (2, 8, 8)) ...)

    -- naming neither sequence parallelism nor the actual cause.

    torch's arrangement works in its own Llama example because the *embedding*
    emits a sequence-sharded activation and the entire residual stream stays
    sharded, so every consumer is written for it.  That is a property of the
    model, not something this framework can install generically: a plain
    ``nn.Linear`` in the middle of such a stream cannot even compute, because
    ``F.linear`` flattens every dimension but the last and DTensor refuses to
    flatten a sharded one.

    What the hand-written ``SequenceParallelLayerNorm`` actually did was much
    narrower, and generically installable: take the local slice of the sequence
    (a local ``chunk`` -- no communication, every rank has the full tensor),
    normalise it, and all-gather the result back.  Only the norm's own
    activation was ever sharded.  This style reproduces exactly that, with
    DTensor placements doing the communication:

    * ``from_local`` on the local chunk, so the global shape is right;
    * the returned DTensor carries ``Shard(sequence_dim)`` through the norm;
    * ``redistribute(Replicate())`` gathers on the way out, and its backward
      reduce-scatters -- which is what the old ``gather_from_sequence`` did by
      hand.

    The replicated parameters' gradients need no ``register_hook``: DTensor
    propagates a Partial gradient for a Replicate parameter fed by a sharded
    activation and reduces it in autograd.  That is the property
    ``test_sequence_parallel_layernorm_gradients_match_dense`` and its
    ``..._reduction_is_load_bearing`` counterfactual check -- the old code did
    this reduction by hand, and without it the copies diverged.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor.parallel import ParallelStyle

    class _ShardedNorm(ParallelStyle):
        def _apply(self, module: Any, device_mesh: Any) -> Any:
            import torch

            def prepare_input(inputs: Any, mesh: Any) -> Any:
                # Describe the scatter as a layout transition and let DTensor's
                # autograd invert it.  Hand-rolling the local `chunk` here is
                # measurably wrong: the chunk's backward hands each rank only
                # its OWN slice's contribution to the input gradient, so with a
                # replicated input every rank ends up with a partially-filled
                # gradient -- measured at world=2 as max|dInput| = 4.8e-03 on
                # one rank and 7.0e-03 on the other, against 9.3e-10 for the
                # unsharded path.  Replicate -> Shard is that same local chunk
                # on the way in, and its autograd is Shard -> Replicate, an
                # all-gather -- which is exactly what the old
                # `scatter_sequence`/`gather_sequence` pair did by hand.
                value = inputs[0]
                replicated = (value if isinstance(value, DTensor) else DTensor.from_local(
                    value, mesh, [Replicate()], run_check=False))
                return (replicated.redistribute(placements=(Shard(sequence_dim),)), *inputs[1:])

            def prepare_output(output: Any, mesh: Any) -> Any:
                if not isinstance(output, DTensor):
                    return output
                return output.redistribute(placements=(Replicate(),)).to_local()

            for name, parameter in module.named_parameters():
                module.register_parameter(
                    name, torch.nn.Parameter(DTensor.from_local(
                        parameter.detach(), device_mesh, [Replicate()], run_check=False)))
            module.register_forward_pre_hook(
                lambda mod, inputs: prepare_input(inputs, device_mesh), with_kwargs=False)
            module.register_forward_hook(lambda mod, inputs, output: prepare_output(output, device_mesh))
            return module

    return _ShardedNorm()


def sequence_parallel_styles(module: Any, *, sequence_dim: int = 1) -> dict[str, Any]:
    """``{fqn: style}`` for every norm the module contains, sharding the sequence.

    See :func:`_sharded_norm_style` for why this is a custom style and not
    torch's ``SequenceParallel``.
    """
    import torch
    style = _sharded_norm_style(sequence_dim)
    # LayerNorm only -- NOT nn.Dropout, which an earlier version wrapped here too.
    #
    # Dropout inside the region draws its mask from the rank's own RNG stream, and
    # every rank's stream starts at the same place.  Rank 0 is dropped out with
    # the masks of the sequence's first block and rank 1 with the SAME masks
    # applied to the second block, so the result is not what one process computes:
    # measured `max|diff| = 9.6e-01` against a single-process reference.
    #
    # PyTorch has the machinery for this (`OffsetBasedRNGTracker` in
    # `distributed.tensor._random`), but it only engages for DTensor random ops
    # and `nn.Dropout` does not go through one, and `is_rng_supported_mesh`
    # returns False on a CPU mesh -- the module says "only supports a GPU device
    # mesh".  So there is no way to make it correct here, and a wrong mask must
    # not be silent.
    #
    # Keeping dropout OUT is what makes it right, not a compromise: this style
    # gathers the norm's output back to the full sequence (see
    # `_sharded_norm_style`), so a dropout that follows a norm sees the whole
    # sequence on every rank and reproduces one process exactly.  Sequence
    # parallelism here shards each norm's own activation, nothing else.
    supported = (torch.nn.LayerNorm,)
    return {name: style for name, child in module.named_modules()
            if name and isinstance(child, supported)}


def parallelize_tensor_parallel(module: Any, *, styles: dict[str, Any], mesh: Any = None,
                                process_group: Any = None, device_type: str = "cpu",
                                src_data_rank: int | None = 0) -> Any:
    """Apply ``styles`` (``{module fqn: torch ParallelStyle}``) over the TP axis.

    Returns ``module``, mutated in place: ``parallelize_module`` replaces each
    named submodule's parameters with DTensors and installs the layout-shifting
    input/output hooks on that submodule.  Nothing about the tree changes, so
    ``state_dict`` keys survive.
    """
    from torch.distributed.tensor.parallel import parallelize_module
    tp_mesh = device_mesh(mesh, process_group, device_type)
    tp_size = int(tp_mesh.size())
    if tp_size > 1 and not styles:
        # The measured trap: tp_size=2 over a plain nn.Sequential left the
        # parameter count unchanged and said nothing.  Every rank would train a
        # full replica, each on its own slice of the data.
        raise TPConfigurationError(
            f"tp_size={tp_size} but the plan names no module to shard, so the model "
            "would run as independent full replicas with no collectives and no error. "
            "Name the projections explicitly (see plugins.transformer.TransformerTPPlan), "
            "or set tp_size=1.")
    if not styles:
        return module
    parallelize_module(module, tp_mesh, styles, src_data_rank=src_data_rank)
    return module
