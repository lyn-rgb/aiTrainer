"""Checkpoint save/load, including the per-stage rank mapping."""

from __future__ import annotations

import os
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


def _stage_prefix(trainer: Any) -> str:
    """The key namespace this rank's checkpoint entries belong to.

    A saved key has to identify one tensor across every rank taking part in the
    collective.  Tensor parallelism does that for free now -- the parameters are
    ``DTensor``, so DCP knows ``q_proj.weight`` is *one* tensor sharded over
    several ranks.  Pipeline parallelism does not: ``split_sequential`` wraps
    each stage's layers in its own ``nn.Sequential``, so every stage names its
    children ``0``, ``1``, ... and stage 0's ``0.weight`` and stage 1's
    ``0.weight`` are different tensors under one key.  DCP keeps one and hands
    it to both, and *both the save and the load report success*.

    Measured at world=2, pp=2: after a save/load round trip each rank held the
    same tensor, neither one its own, with max|dW| = 6.4e-01 against its trained
    weights.  Making the keys unique per stage -- nothing else -- gives an exact
    round trip (0.000e+00).

    So the prefix is required exactly when there is more than one stage, which
    is a statement about key uniqueness rather than a configuration switch: with
    one stage, ``0.weight`` is already unique and the unprefixed layout is kept,
    so single-stage checkpoints are unchanged.
    """
    parallel = getattr(getattr(trainer, "config", None), "parallel", None)
    if int(getattr(parallel, "pp_size", 1) or 1) <= 1:
        return ""
    stage = getattr(getattr(trainer, "model", None), "module", None)
    return f"stage.{int(getattr(stage, 'stage_id', 0) or 0)}."


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


def save_sharded(trainer: Any, path: str | os.PathLike[str]) -> None:
    """Write a sharded checkpoint with ``torch.distributed.checkpoint``.

    Opt-in, and deliberately separate from :func:`save_checkpoint`: that one
    writes a complete ``rank_state.pt`` on every rank, which is simple and
    well-tested but costs ``world_size`` copies of the model.  The two formats
    are not interchangeable, so they are two entry points rather than one
    function whose behaviour depends on configuration.

    DCP is collective, so every rank must call this with the SAME path.
    """
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )
    except ImportError as exc:
        raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
    import torch

    module = _stateful_module(trainer.model)
    trainer.overlap.drain(trainer.config.overlap.drain_timeout_s)
    trainer.offload.drain(trainer.optimizer)
    prefix = _stage_prefix(trainer)
    state = {
        f"{prefix}model": get_model_state_dict(module),
        f"{prefix}optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        f"{prefix}bookkeeping": torch.tensor([trainer.global_step, trainer.optimizer_step],
                                             dtype=torch.int64),
    }
    CheckpointManager(runtime=trainer.runtime).save_dcp(
        path, state=state,
        metadata={"world_size": trainer.runtime.world_size, "format": "dcp-sharded",
                  "global_step": trainer.global_step, "optimizer_step": trainer.optimizer_step})


def load_sharded(trainer: Any, path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read back what :func:`save_sharded` wrote, on every rank.

    The state dict passed in is a *recipe*: DCP fills it in place, reading each
    rank's own shards.  Bookkeeping travels as a tensor so it survives the same
    path as everything else.
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
    prefix = _stage_prefix(trainer)
    state = {
        f"{prefix}model": get_model_state_dict(module),
        f"{prefix}optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        f"{prefix}bookkeeping": torch.zeros(2, dtype=torch.int64),
    }
    metadata = CheckpointManager(runtime=trainer.runtime).load_dcp(path, state=state)
    set_model_state_dict(module, state[f"{prefix}model"])
    set_optimizer_state_dict(module, trainer.optimizer, state[f"{prefix}optimizer"])
    trainer.global_step = int(state[f"{prefix}bookkeeping"][0].item())
    trainer.optimizer_step = int(state[f"{prefix}bookkeeping"][1].item())
    return {"global_step": trainer.global_step, "optimizer_step": trainer.optimizer_step,
            "metadata": metadata}
