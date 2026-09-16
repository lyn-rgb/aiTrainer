"""Uniform wrapper for the ordinary (non-pipeline) model path."""

from __future__ import annotations

from typing import Any, Iterable

from .core.batching import call_with_inputs, compute_loss
from .parallel.fsdp import no_gradient_sync
from .parallel.pp_schedule import ScheduleOutput


class DistributedModel:
    """A deliberately thin Batch 0 wrapper around ``torch.nn.Module``."""

    def __init__(self, module: Any) -> None:
        self.module = module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True) -> "DistributedModel":
        self.module.train(mode)
        return self

    def eval(self) -> "DistributedModel":
        return self.train(False)

    def parameters(self, recurse: bool = True) -> Iterable[Any]:
        return self.module.parameters(recurse=recurse)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any], **kwargs: Any) -> Any:
        return self.module.load_state_dict(state, **kwargs)

    def no_sync(self):
        """Defer the gradient reduction; see :func:`parallel.fsdp.no_gradient_sync`.

        This used to be ``getattr(self.module, "no_sync", nullcontext)()``, which
        silently degraded to a no-op for anything without that exact method --
        including every FSDP2 module, since FSDP2 renamed it.  The result would
        have been a full gradient reduction per microbatch instead of one per
        accumulation window: slower, and no exception to notice it by.
        """
        if hasattr(self.module, "no_sync"):
            return self.module.no_sync()
        return no_gradient_sync(self.module)


class PipelineStage:
    """A stage module plus immutable stage metadata and schedule ownership."""

    def __init__(self, module: Any, *, stage_id: int,
                 schedule: Any = None, pp_group: Any = None, pp_size: int = 1,
                 pp_ranks: tuple[int, ...] | None = None, num_microbatches: int = 1,
                 tp_rank: int = 0, dp_rank: int = 0) -> None:
        self.module = module
        self.stage_id = stage_id
        self.schedule = schedule
        self.pp_group = pp_group
        self.pp_size = pp_size
        self.pp_ranks = pp_ranks
        self.num_microbatches = num_microbatches
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterable[Any]:
        return self.module.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> "PipelineStage":
        self.module.train(mode)
        return self

    def eval(self) -> "PipelineStage":
        return self.train(False)

    def no_sync(self):
        if hasattr(self.module, "no_sync"):
            return self.module.no_sync()
        return no_gradient_sync(self.module)

    # No clip_grad_norm_ method here.  It existed, had no callers, and its
    # fallback was torch.nn.utils -- which under FSDP2 silently returns a
    # shard-local norm (see parallel.fsdp.clip_grad_norm_).  The trainer clips
    # through that function with this stage as the module, so a second route to
    # the same operation could only drift.

    def to(self, *args: Any, **kwargs: Any) -> "PipelineStage":
        self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any], **kwargs: Any) -> Any:
        return self.module.load_state_dict(state, **kwargs)

    @property
    def uses_pipeline(self) -> bool:
        return self.pp_size > 1

    def _device(self) -> Any:
        import torch
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    def _schedule_for(self, *, loss_fn: Any, scaler: Any = None,
                      accumulation_steps: int = 1) -> Any:
        """Build the schedule this stage runs.

        The only place a schedule NAME becomes a schedule OBJECT.  ``schedule``
        used to be a string on the production path and an object in
        ``pp_schedule``'s own consumers, and the two conventions were reconciled
        by string comparison at every use.
        """
        from .parallel.pp_p2p import P2PCommunicator
        from .parallel.pp_schedule import build_schedule
        communicator = P2PCommunicator(pp_rank=self.stage_id, pp_size=self.pp_size,
                                       pp_group=self.pp_group, pp_ranks=self.pp_ranks)
        name = self.schedule if isinstance(self.schedule, str) else "gpipe"
        return build_schedule(name, stage_module=self.module, communicator=communicator,
                              num_microbatches=self.num_microbatches, loss_fn=loss_fn,
                              stage_id=self.stage_id, scaler=scaler,
                              accumulation_steps=accumulation_steps, device=self._device())

    def pipeline_step(self, batch: Any, *, loss_fn: Any = None, scaler: Any = None,
                      accumulation_steps: int = 1) -> Any:
        """Run one pipeline step for the current stage.

        At ``pp_size > 1`` this delegates to the schedule, which owns the
        forward/backward ordering, the metadata protocol and the window
        denominator.  This class used to carry three copies of those algorithms;
        the copies are gone, so the 17 in-process protocol tests now guard the
        code that actually runs.
        """
        if self.pp_size <= 1:
            output = call_with_inputs(self.module, batch)
            loss = compute_loss(output, batch, loss_fn)
            scaled = loss / max(1, accumulation_steps)
            (scaler.scale(scaled) if scaler is not None and scaler.is_enabled() else scaled).backward()
            return ScheduleOutput(loss.detach(), (output,), 1, True)

        schedule = self._schedule_for(loss_fn=loss_fn, scaler=scaler,
                                      accumulation_steps=accumulation_steps)
        return schedule.run_stage(batch)

    def pipeline_evaluate(self, batch: Any, *, loss_fn: Any = None) -> Any:
        """Forward-only PP traversal used by ``Trainer.evaluate``."""
        schedule = self._schedule_for(loss_fn=loss_fn)
        return schedule.evaluate(batch, loss_fn=loss_fn)
