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

    The replicated parameters' gradients need reducing by hand, and an earlier
    version of this docstring claimed the opposite -- that DTensor "propagates a
    Partial gradient for a Replicate parameter fed by a sharded activation and
    reduces it in autograd".  Measured: it propagates the Partial and never
    reduces it.  At world=2 the norm parameters' ``.grad`` came back
    ``(Partial(sum),)`` with rank-local values that differ, while a Replicate
    parameter fed by a *Replicate* activation (a row projection's bias) comes
    back ``(Replicate(),)`` and agrees.  So the placement is decided by what
    feeds the module, not normalised on the way out.

    That left three symptoms, and the quiet ones are the dangerous ones:

    * the parameter *update* was still correct, because DTensor reduces when it
      computes the new parameter value, so the loss curve and the weights after
      a step both look right;
    * the *optimizer state* was not: SGD's momentum buffer ends up
      ``(Partial(sum),)`` holding each rank's own partial sum, so the ranks
      store different momentum.  A sharded checkpoint then has one rank's
      partial sum to write, and restoring it gave every rank that same wrong
      buffer -- a resumed run diverged by 2.08e-01 while the write/read of the
      weights themselves stayed bit-for-bit exact;
    * anything reading ``.grad`` directly was wrong, gradient clipping included,
      since it would clip the norm of a partial sum.

    The reduction is :func:`reduce_replicated_gradients`, called from the
    optimizer step rather than installed here as a tensor hook.  A
    ``register_post_accumulate_grad_hook`` on these parameters looks like the
    natural home and does work -- measured, the gradient came back
    ``(Replicate(),)`` agreeing across ranks.  It is wrong anyway, and silently
    so: **one call to ``model.to(device)`` permanently stops the hook firing**
    on these DTensor parameters.  ``nn.Module.to`` repoints ``.data``, the hook
    stays listed on the parameter, ``.grad`` still gets populated -- with the
    unreduced Partial -- and nothing raises.  ``Trainer.__init__`` calls
    ``.to()`` on every model it wraps, so a hook installed here would be dead in
    the framework's own path while looking correct in isolation.  Reducing where
    the gradient is CONSUMED instead needs no assumption about who has moved the
    model, and costs one collective per optimizer step rather than one per
    microbatch.
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
    return {name: style for name, child in module.named_modules()
            if name and isinstance(child, sequence_parallel_types())}


def sequence_parallel_types() -> tuple[Any, ...]:
    """The module types the SP style can shard, in one place.

    Matching structurally rather than by name is what keeps sequence
    parallelism from imposing a naming convention on the model -- but it also
    means any OTHER norm type matches nothing at all, which
    :func:`require_sequence_parallel_targets` exists to turn into an error
    instead of a silent no-op.

    ``nn.RMSNorm`` is included because the style is indifferent to which norm it
    wraps: it shards the **sequence** axis and leaves the norm's own axis -- the
    last one -- whole on every rank, so it applies to anything normalising over
    the hidden dimension.  Measured against a dense reference at world=2, on a
    block whose norm is ``nn.RMSNorm``: forward, input gradient and norm
    parameter gradient all agree (see
    ``test_sequence_parallel_matches_dense_rmsnorm``).  It is added only when
    the installed torch has it, since ``nn.RMSNorm`` arrived in 2.4.
    """
    import torch

    rms_norm = getattr(torch.nn, "RMSNorm", None)
    return (torch.nn.LayerNorm,) if rms_norm is None else (torch.nn.LayerNorm, rms_norm)


def require_sequence_parallel_targets(module: Any) -> None:
    """Refuse sequence parallelism over a model that has no norm it can shard.

    Measured before this guard existed, on a model using a norm type outside
    ``sequence_parallel_types()`` at ``tp_size=2`` with ``sp_backend='megatron'``:
    the TP plan matched the projections, so ``parallelize_tensor_parallel`` had
    no reason to complain, and ``sequence_parallel_styles`` returned ``{}`` --
    the run then trained with sequence parallelism configured, reported, and not
    applied.  Every norm's weight stayed a plain ``Parameter``.  The loss curve
    is unaffected (SP is an activation-layout optimisation), so nothing surfaces
    it.  ``nn.RMSNorm`` was that case once; it is now covered rather than
    refused, which is what the guard's message tells the next one to do.

    This is the same failure ``parallelize_tensor_parallel`` already refuses for
    TP -- "the plan names no module to shard" -- and it was only guarded on that
    side.  A model with no recognised norm anywhere has nothing for SP to do on
    ANY stage, so refusing is never rejecting a working configuration.

    Checked against the **whole model**, not this rank's stage: under pipeline
    parallelism a stage legitimately holds no norm (an embedding-only first
    stage, say), and refusing that would reject a working split.  Every rank
    reaches the same verdict because every rank sees the same unsplit model, so
    the failure is a clean error on all ranks rather than a hang on one.
    """
    for _, child in module.named_modules():
        if isinstance(child, sequence_parallel_types()):
            return
    raise TPConfigurationError(
        "parallel.sp_backend is set but this model contains no norm the "
        "sequence-parallel style can shard (it knows "
        f"{', '.join(sorted(item.__name__ for item in sequence_parallel_types()))}) "
        "-- so it would run with sequence parallelism configured and not applied, "
        "and nothing would report it. A model using another norm type needs that "
        "type added to parallel.tp.sequence_parallel_types(), or set "
        "sp_backend='none'.")


def reduce_replicated_gradients(module: Any) -> int:
    """All-reduce every replicated parameter's still-Partial gradient, in place.

    :func:`_sharded_norm_style` hands the module a ``Shard(sequence_dim)``
    activation, so the norm's ``Replicate`` parameters accumulate a
    ``(Partial(sum),)`` gradient -- each rank holding its own partial sum -- and
    nothing reduces it.  Returns how many parameters were reduced.

    Called from ``trainer.step.run_optimizer_step`` before clipping and
    ``optimizer.step``, because a tensor hook cannot be used: see
    :func:`_sharded_norm_style` for the measured reason (``model.to()`` silently
    disables one).  Reducing at the point of consumption also means once per
    optimizer step instead of once per microbatch, and it fixes gradient
    clipping, which would otherwise take the norm of a partial sum.

    Only two things are touched, and both conditions have to hold:

    * the **parameter** is entirely replicated, so the sum over ranks is what it
      should hold.  A sharded parameter's gradient is a shard, never a Partial,
      and reducing one would produce a full tensor for a shard-shaped
      parameter;
    * the **gradient** is entirely Partial.  A mixed placement (``Partial`` on
      one mesh axis and ``Shard`` on another) is not a case this framework
      produces, and guessing at it would be worse than leaving it alone.

    Anything else is untouched, so on the ordinary all-Replicate path this
    walks the parameters and does nothing.
    """
    from torch.distributed.tensor import DTensor, Replicate

    reduced = 0
    for parameter in module.parameters():
        placements = getattr(parameter, "placements", None)
        if placements is None or not all(item.is_replicate() for item in placements):
            continue
        gradient = parameter.grad
        if not isinstance(gradient, DTensor):
            continue
        if not gradient.placements or not all(item.is_partial() for item in gradient.placements):
            continue
        parameter.grad = gradient.redistribute(
            placements=tuple(Replicate() for _ in gradient.placements))
        reduced += 1
    return reduced


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
