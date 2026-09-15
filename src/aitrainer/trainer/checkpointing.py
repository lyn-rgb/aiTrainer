"""Checkpoint save/load, including the per-stage rank mapping."""

from __future__ import annotations

import os
from typing import Any

from ..checkpoint.manager import CheckpointManager


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
