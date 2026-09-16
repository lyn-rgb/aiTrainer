"""Checkpoint save/load, including the per-stage rank mapping."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..checkpoint.manager import CheckpointError, CheckpointManager


def save_checkpoint(trainer: Any, path: str | os.PathLike[str]) -> None:
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    rank_mapping = {"rank": trainer.runtime.rank, "world_size": trainer.runtime.world_size}
    module = trainer.model.module
    for attribute, key in (("stage_id", "pp_rank"), ("tp_rank", "tp_rank"), ("dp_rank", "dp_rank"),
                           ("pp_size", "pp_size")):
        if hasattr(module, attribute):
            rank_mapping[key] = int(getattr(module, attribute))
    CheckpointManager(runtime=trainer.runtime).save(
        path, model=trainer.model, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
        scaler=trainer.scaler, global_step=trainer.global_step, optimizer_step=trainer.optimizer_step,
        config=trainer.config.to_dict(), rank_mapping=rank_mapping,
        sampler_state=(trainer._data_provider.state_dict() if trainer._data_provider is not None else None))


def load_checkpoint(trainer: Any, path: str | os.PathLike[str]) -> dict[str, Any]:
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    state = CheckpointManager(runtime=trainer.runtime).load(
        path, model=trainer.model, optimizer=trainer.optimizer, scheduler=trainer.scheduler,
        scaler=trainer.scaler, expected_world_size=trainer.runtime.world_size)
    trainer.global_step = int(state["global_step"])
    trainer.optimizer_step = int(state["optimizer_step"])
    return state


def _stateful_module(model: Any) -> Any:
    """The object torch's distributed state-dict helpers should be handed.

    ``Trainer.model`` is a ``DistributedModel``; underneath it may be a plain
    module, a ``PipelineStage``, or an FSDP2-sharded module (which *is* the module
    -- ``fully_shard`` mutates in place, so there is no wrapper to unwrap).
    ``get_model_state_dict`` understands FSDP and DDP directly.
    """
    inner = getattr(model, "module", model)
    if hasattr(inner, "stage_id"):          # PipelineStage: the stage owns the module
        return inner.module
    return inner


def _coordinate(trainer: Any) -> tuple[int, int, int]:
    """This rank's ``(pp, tp, dp)``, from the mesh ``parallelize`` built."""
    mesh = getattr(getattr(trainer, "runtime", None), "mesh", None)
    if mesh is None:
        return (0, 0, 0)
    coordinate = mesh.coordinate
    return (int(coordinate.pp), int(coordinate.tp), int(coordinate.dp))


def _rng_state() -> dict[str, Any]:
    """The process RNG state, as fixed-shape tensors.

    DCP decides how many bytes to read from the SHAPE of the state dict it is
    handed, so the shape has to be the same on the way out and the way back in.
    Pickling Python's state does not give that: ``random.getstate()`` carries a
    cached gaussian that is ``None`` until a gaussian draw happens, and the
    pickle grows when it becomes a float -- measured 3752 bytes on save against
    3829 on load, which DCP reports as "Size mismatch between saved
    torch.Size([3752]) and current torch.Size([3829])".

    So the state is decomposed instead of pickled: a version, the Mersenne
    Twister words, and the cached gaussian with NaN standing in for "none".
    """
    import random

    import torch
    version, internal, gauss = random.getstate()
    state = {
        "torch": torch.get_rng_state(),
        "python_version": torch.tensor([int(version)], dtype=torch.int64),
        "python_internal": torch.tensor([int(word) for word in internal], dtype=torch.int64),
        "python_gauss": torch.tensor([float("nan") if gauss is None else float(gauss)],
                                     dtype=torch.float64),
    }
    if torch.cuda.is_available():
        # The per-rank format has always stored these (``_torch_rng_state`` in
        # checkpoint/manager.py); this path stored only the CPU generator, so on
        # a GPU box a resumed run drew different dropout masks than the run it
        # resumed -- and a CPU-only host cannot see it, because
        # ``is_available()`` is False and there is no CUDA generator to miss.
        #
        # Stacked into one (devices, bytes) tensor: every device's state is the
        # same length for a given torch build, so this keeps the fixed shape DCP
        # needs, and the device count is the same on the save and the load of one
        # checkpoint.
        state["cuda"] = torch.stack([item.cpu() for item in torch.cuda.get_rng_state_all()])
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    import math
    import random

    import torch
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([row.cpu() for row in state["cuda"]])
    gauss = float(state["python_gauss"][0])
    random.setstate((int(state["python_version"][0]),
                     tuple(int(word) for word in state["python_internal"].tolist()),
                     None if math.isnan(gauss) else gauss))


def _restore_missing_group_keys(optimizer: Any, saved: list[dict[str, Any]]) -> list[str]:
    """Put back any optimizer hyperparameter the load dropped.

    ``set_optimizer_state_dict`` adopts the checkpoint's ``param_groups``, and DCP
    does not always fill every non-tensor leaf of them.  Measured on a two-stage
    pipeline with SGD+momentum: ``load_sharded`` returned successfully and rank
    0's group came back without ``dampening``/``lr``/``momentum``/``weight_decay``
    while rank 1's came back whole -- the loss was never taken, and the failure
    surfaced on the NEXT step as ``KeyError: 'momentum'`` (``'betas'`` with AdamW)
    from inside ``torch.optim``, which names the key but not where it went.

    ``setdefault`` only fills what is absent, so a checkpoint that does carry a
    value keeps it.  The group count is checked because a mismatch would silently
    misalign which group is which.
    """
    restored: list[str] = []
    groups = optimizer.param_groups
    if len(groups) != len(saved):
        raise CheckpointError(
            f"optimizer has {len(groups)} parameter groups but the checkpoint was taken "
            f"with {len(saved)}; refusing to guess which is which")
    for current, previous in zip(groups, saved):
        for key, value in previous.items():
            if key not in current:
                current[key] = value
                restored.append(key)
    return restored


def _object_state(obj: Any) -> dict[str, Any] | None:
    """``obj.state_dict()``, or None when the trainer has no such object.

    None rather than ``{}`` on purpose: an empty dict writes no keys, so a
    restore that *does* have the object fails loudly on a missing key instead of
    quietly restoring nothing.  The distinction is the whole reason this is a
    function and not an inline call.
    """
    if obj is None or not hasattr(obj, "state_dict"):
        return None
    return obj.state_dict()


# The data provider is only known INSIDE ``fit`` (``loop.fit`` assigns
# ``trainer._data_provider`` from its ``data`` argument), so at ``load_sharded``
# time there is usually no live object to derive a placeholder shape from.
# Its state therefore travels as bytes, a non-tensor DCP entry: DCP replaces
# non-tensor values wholesale, so any placeholder works and no shape has to be
# predicted.  (A tensor would have to match the saved shape exactly -- the trap
# ``_rng_state`` exists to avoid.)  ``fit`` applies it on the next call.
_SAMPLER_KEY = "sampler"


def save_sharded(trainer: Any, path: str | os.PathLike[str]) -> None:
    """Write a sharded checkpoint with ``torch.distributed.checkpoint``.

    Opt-in, and deliberately separate from :func:`save_checkpoint`: that one
    writes a complete ``rank_state.pt`` on every rank, which is simple and
    well-tested but costs ``world_size`` copies of the model.  The two formats
    are not interchangeable, so they are two entry points rather than one
    function whose behaviour depends on configuration.

    DCP is collective, so every rank must call this with the SAME path.

    This carries the same contents as :func:`save_checkpoint` -- model,
    optimizer, scheduler, scaler, sampler position, step counters and both RNG
    streams.  It used to write only model/optimizer/bookkeeping, so a run resumed
    from it diverged from the run that wrote it: no RNG (fixed earlier), and no
    scheduler, so the LR schedule restarted from step 0.  Measured with
    ``CosineAnnealingLR``: after four steps the reference run sits at lr
    0.065451, and resuming two more steps from a sharded checkpoint gave
    0.059201 against the correct 0.034549.  ``StepLR`` hides this, because it
    multiplies the current lr (which the optimizer state restores) rather than
    computing one from ``last_epoch``.
    """
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )
    except ImportError as exc:
        raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
    import pickle

    import torch

    module = _stateful_module(trainer.model)
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    # The RNG state is per-rank and the ranks genuinely differ (data parallelism
    # gives each one different batches), so it cannot share a key: DCP would read
    # one rank's version as THE value and hand it to everybody.  The coordinate
    # namespaces it, the same way PP stages need their own namespace.
    coordinate = _coordinate(trainer)
    state: dict[str, Any] = {
        "model": get_model_state_dict(module),
        "optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        "bookkeeping": torch.tensor([trainer.global_step, trainer.optimizer_step],
                                    dtype=torch.int64),
        f"rng.{coordinate[0]}.{coordinate[1]}.{coordinate[2]}": _rng_state(),
    }
    for key, obj in (("scheduler", trainer.scheduler), ("scaler", trainer.scaler)):
        payload = _object_state(obj)
        if payload is not None:
            state[key] = payload
    provider = getattr(trainer, "_data_provider", None)
    if provider is not None and hasattr(provider, "state_dict"):
        state[_SAMPLER_KEY] = pickle.dumps(provider.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    CheckpointManager(runtime=trainer.runtime).save_dcp(
        path, state=state,
        metadata={"world_size": trainer.runtime.world_size, "format": "dcp-sharded",
                  "global_step": trainer.global_step, "optimizer_step": trainer.optimizer_step})


def load_sharded(trainer: Any, path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read back what :func:`save_sharded` wrote, on every rank.

    The state dict passed in is a *recipe*: DCP fills it in place, reading each
    rank's own shards.  Bookkeeping travels as a tensor so it survives the same
    path as everything else.

    Scheduler and scaler states carry their own object's layout as the
    placeholder, so a checkpoint written by a different kind of scheduler is
    refused by DCP rather than applied on top of the wrong one.  The sampler
    position is left on the trainer for the next :meth:`Trainer.fit` to apply --
    see ``_SAMPLER_KEY``.
    """
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
        )
    except ImportError as exc:
        raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
    import torch

    module = _stateful_module(trainer.model)
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    coordinate = _coordinate(trainer)
    rng_key = f"rng.{coordinate[0]}.{coordinate[1]}.{coordinate[2]}"
    state: dict[str, Any] = {
        "model": get_model_state_dict(module),
        "optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        "bookkeeping": torch.zeros(2, dtype=torch.int64),
        # NOT a stand-in: DCP reads the shape of the state dict it is handed to
        # decide how much to read, so this has to be the same SHAPE as what was
        # saved.  A zero-length placeholder failed with "Size mismatch between
        # saved torch.Size([5056]) and current: torch.Size([0])", and the
        # placeholder has to be built from the current state rather than a
        # constant.  ``_rng_state`` is fixed-shape by construction -- that is
        # what it is for; see its docstring.
        rng_key: _rng_state(),
    }
    for key, obj in (("scheduler", trainer.scheduler), ("scaler", trainer.scaler)):
        payload = _object_state(obj)
        if payload is not None:
            state[key] = payload
    # Only claimed when the checkpoint actually holds it: DCP refuses a recipe
    # key that is not in the checkpoint, and a provider is optional, so adding
    # this unconditionally would break every run saved without one.
    manager = CheckpointManager(runtime=trainer.runtime)
    if _SAMPLER_KEY in manager.dcp_keys(path):
        state[_SAMPLER_KEY] = b""
    metadata = manager.load_dcp(path, state=state)
    _restore_rng(state[rng_key])
    set_model_state_dict(module, state["model"])
    group_defaults = [dict(group) for group in trainer.optimizer.param_groups]
    set_optimizer_state_dict(module, trainer.optimizer, state["optimizer"])
    _restore_missing_group_keys(trainer.optimizer, group_defaults)
    if trainer.scheduler is not None and state.get("scheduler") is not None:
        trainer.scheduler.load_state_dict(dict(state["scheduler"]))
    if trainer.scaler is not None and state.get("scaler") is not None:
        trainer.scaler.load_state_dict(dict(state["scaler"]))
    # Stashed rather than applied: the provider is not known until fit() is
    # called with the data, and a caller resuming a run calls load_sharded()
    # first.  fit() consumes this and clears it.
    trainer._restored_sampler_state = state.get(_SAMPLER_KEY)
    trainer.global_step = int(state["bookkeeping"][0].item())
    trainer.optimizer_step = int(state["bookkeeping"][1].item())
    return {"global_step": trainer.global_step, "optimizer_step": trainer.optimizer_step,
            "sampler_state": state.get(_SAMPLER_KEY), "metadata": metadata}


def load_pretrained(trainer: Any, path: str | os.PathLike[str], *,
                    base_dir: str | os.PathLike[str] | None = None,
                    map_location: Any = "cpu") -> dict[str, Any]:
    """Load a converted pretrained checkpoint, each rank reading only its own shards.

    The model must already be parallelised: ``load_for_rank`` writes into the
    parameters the ranks actually hold, so under FSDP each rank reads its own
    slice of each tensor and nothing is broadcast.  That is the whole point --
    the alternative is one rank reading the complete model and shipping it out.

    ``path`` is a manifest produced by ``CheckpointConverter.convert``.  Converting
    a dense checkpoint is an offline step on purpose: reading a single-file
    checkpoint means reading all of it, so it must not happen once per rank at
    every startup.  Point ``convert(dp_sharded=...)`` at the layout the training
    run will use.
    """
    from ..checkpoint import ModelLoader
    from ..mesh import DeviceMeshManager

    parallel = trainer.config.parallel
    runtime = trainer.runtime
    mesh = DeviceMeshManager(
        pp_size=parallel.pp_size, dp_size=parallel.dp_size, tp_size=parallel.tp_size,
        world_size=runtime.world_size,
        device_type=str(runtime.state.device).split(":", 1)[0])
    coordinate = mesh.coordinate
    # FSDP shards the DP axis; without it the axis replicates and the checkpoint
    # must hold copies.
    dp_sharded = bool(trainer.config.fsdp.enabled and parallel.dp_size > 1)
    module = _stateful_module(trainer.model)
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    stats = ModelLoader().load_for_rank(
        path, module.state_dict(),
        pp_rank=coordinate.pp, tp_rank=coordinate.tp, dp_rank=coordinate.dp,
        logical_sharding={"pp": parallel.pp_size, "tp": parallel.tp_size, "dp": parallel.dp_size},
        dp_sharded=dp_sharded, base_dir=base_dir, map_location=map_location)
    return stats
