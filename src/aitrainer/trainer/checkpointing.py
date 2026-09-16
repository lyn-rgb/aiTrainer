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
    state = {
        "model": get_model_state_dict(module),
        "optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        "bookkeeping": torch.tensor([trainer.global_step, trainer.optimizer_step],
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
    state = {
        "model": get_model_state_dict(module),
        "optimizer": get_optimizer_state_dict(module, trainer.optimizer),
        "bookkeeping": torch.zeros(2, dtype=torch.int64),
    }
    metadata = CheckpointManager(runtime=trainer.runtime).load_dcp(path, state=state)
    set_model_state_dict(module, state["model"])
    set_optimizer_state_dict(module, trainer.optimizer, state["optimizer"])
    trainer.global_step = int(state["bookkeeping"][0].item())
    trainer.optimizer_step = int(state["bookkeeping"][1].item())
    return {"global_step": trainer.global_step, "optimizer_step": trainer.optimizer_step,
            "metadata": metadata}
