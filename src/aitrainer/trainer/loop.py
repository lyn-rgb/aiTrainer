"""The epoch / step / resume loop."""

from __future__ import annotations

import logging
import os

from .state import StepOutput

logger = logging.getLogger("aitrainer")


def fit(trainer, data, *, epochs: int = 1, max_steps: int | None = None,
        resume_from: str | os.PathLike[str] | None = None) -> list[StepOutput]:
    trainer._data_provider = data if hasattr(data, "state_dict") else None
    if resume_from is not None:
        state = trainer.load_checkpoint(resume_from)
        if trainer._data_provider is not None and state.get("sampler_state"):
            trainer._data_provider.load_state_dict(state["sampler_state"])
    for epoch in range(epochs):
        if hasattr(data, "set_epoch"):
            data.set_epoch(epoch)
        loader = (data.train_dataloader(dp_group=None, seed=trainer.config.seed)
                  if hasattr(data, "train_dataloader") else data)
        logger.info("starting epoch with global_step=%d", trainer.global_step)
        for batch in loader:
            trainer.train_step(batch)
            if trainer.global_step % trainer.config.log_interval == 0:
                logger.info("step=%d loss=%.6f", trainer.global_step, trainer.history[-1].loss)
            if max_steps is not None and trainer.global_step >= max_steps:
                # Drop the partial accumulation: leaving it in .grad let a
                # later fit()/train_step() fold it into its first update.
                trainer.optimizer.zero_grad(set_to_none=True)
                return trainer.history
    # A trailing partial accumulation window is DROPPED, not flushed.  A flush
    # cannot be made correct: the tail microbatches were accumulated under
    # no_sync(), so with dp_size>1 their gradients were never reduced -- a
    # synced backward is what materialises the reduction, and there is no
    # forward left to re-run -- and they are scaled k/G where the window mean
    # is k/k.  Stepping on them both diverged the DP replicas and mis-scaled
    # the update; with nothing accumulated at all it advanced the LR schedule
    # for an update that never happened.  Zeroing keeps the tail out of any
    # later fit()/train_step(), matching the max_steps path above.
    trainer.optimizer.zero_grad(set_to_none=True)
    return trainer.history
