"""The ``Trainer`` facade: state, construction, and delegation."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from ..adapters import LossFn
from ..capability import validate_capabilities
from ..config import FrameworkConfig
from ..core.batching import compute_loss
from ..core.torch import require_torch
from ..distributed_model import DistributedModel
from ..offload import OffloadManager
from ..overlap import OverlapController
from ..precision import autocast_context, validate_precision
from ..runtime import Runtime
from . import bootstrap, checkpointing, evaluation, loop, step
from .state import StepOutput

logger = logging.getLogger("aitrainer")


class Trainer:
    """Train a regular PyTorch module with AMP, accumulation and checkpoints."""

    def __init__(self, model: Any, optimizer: Any, *, config: FrameworkConfig | None = None,
                 loss_fn: LossFn | Callable[[Any, Any], Any] | None = None,
                 scheduler: Any | None = None, device: str | None = None,
                 runtime: Runtime | None = None) -> None:
        """Wrap an already-built model.

        The seed is applied when the runtime is created, which is here -- so a
        model the caller built beforehand has already drawn its initialisation
        from the ambient RNG.  For reproducible weight init use
        :meth:`from_model`, or create a :class:`Runtime` (which seeds) before
        building the model.
        """
        torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
        self.config = config or FrameworkConfig()
        self.runtime = runtime or Runtime(device=device or self.config.device, seed=self.config.seed)
        # Capability gating must not depend on the entry point.  Trainer.from_model
        # routes through parallelize() (which validates), but constructing a Trainer
        # directly skipped it -- so FSDP + gradient-bucket overlap was accepted
        # silently here while dry_run/from_model rejected the same config.
        # validate_capabilities runs config.validate itself, so the separate call
        # that used to sit above this line only validated everything twice.
        validate_capabilities(self.config, world_size=self.runtime.world_size, torch_module=torch)
        if self.runtime.world_size > 1 and self.config.parallel.dp_size > 1:
            # Only REPLICATED ranks need to reduce gradients together.  TP/PP with
            # dp_size == 1 has no replicas to diverge, and TP already reduces in the
            # autograd collectives -- demanding no_sync there made the STABLE `tp`
            # capability impossible to train through Trainer.  Inspect the layer
            # that actually reduces, not the wrapper: DistributedModel.no_sync()
            # degrades to nullcontext() when the module it wraps has none, so
            # testing the wrapper let an unsynchronised model pass this guard.
            wrapped = model.module if isinstance(model, DistributedModel) else model
            if not hasattr(wrapped, "no_sync"):
                raise ValueError(
                    "dp_size>1 requires a model that reduces gradients (an FSDP-wrapped "
                    "module): the given model has no no_sync, so each rank would step on "
                    "its own divergent gradients. Set fsdp.enabled=True, or use dp_size=1 "
                    "with tp_size/pp_size for the non-replicated axes.")
        self.model = model if isinstance(model, DistributedModel) else DistributedModel(model)
        self.device = torch.device(device or self.runtime.state.device)
        validate_precision(self.config.precision, device=self.device, torch_module=torch)
        self.model.module.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        use_scaler = self.config.precision.use_grad_scaler
        compute_name = self._dtype_name(self.config.precision.compute_dtype)
        # A reduced-precision compute dtype only takes effect on CUDA (see
        # _autocast).  On CPU it silently runs fp32; previously the only check was
        # an unreachable branch in capability.py, so nothing told the user.
        if compute_name in {"float16", "bfloat16"} and self.device.type != "cuda":
            warnings.warn(
                f"precision.compute_dtype={compute_name!r} has no effect on device "
                f"{self.device.type!r}; training runs in fp32",
                RuntimeWarning, stacklevel=2)
        if use_scaler is None:
            use_scaler = compute_name == "float16"
        enabled = bool(use_scaler and self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        self.global_step = 0
        self.optimizer_step = 0
        self.history: list[StepOutput] = []
        self._data_provider: Any = None
        self.offload = OffloadManager(self.config.offload, overlap_config=self.config.overlap)
        # No `enabled=` flag: the five switches this used to fold together only
        # fed a stored-and-never-read attribute, so the controller behaved
        # identically either way.  The budgets below are the part that is read.
        self.overlap = OverlapController(max_inflight_ops=self.config.overlap.max_inflight_ops,
                                         max_inflight_bytes=self.config.overlap.max_inflight_bytes)
        self.offload.register_model(self.model.module)
        self.offload.register_optimizer(self.optimizer)

    @classmethod
    def from_model(cls, model_or_adapter: Any, *, config: FrameworkConfig | None = None,
                   optimizer: Any | None = None, optimizer_factory: Any | None = None,
                   optimizer_cls: Any | None = None, optimizer_kwargs: Mapping[str, Any] | None = None,
                   loss_fn: LossFn | Callable[[Any, Any], Any] | None = None,
                   scheduler: Any | None = None, device: str | None = None,
                   runtime: Runtime | None = None) -> Trainer:
        """Build a trainer from a model or adapter without hiding parallel choices."""
        model, optimizer, runtime_obj = bootstrap.prepare(
            model_or_adapter, config=config, optimizer=optimizer,
            optimizer_factory=optimizer_factory, optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs, device=device, runtime=runtime)
        return cls(model, optimizer, config=config, loss_fn=loss_fn, scheduler=scheduler,
                   device=device, runtime=runtime_obj)

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        return str(dtype).replace("torch.", "").lower()

    def _autocast(self):
        # `precision.autocast_context` is the single implementation: this method
        # used to be a second copy that mapped the dtype by hand and fell back to
        # the deprecated `torch.cuda.amp.autocast`.  Keep the private name only
        # because three call sites read it.
        return autocast_context(self.config.precision, self.device)

    def _compute_loss(self, output: Any, batch: Any) -> Any:
        # One implementation, shared with the pipeline paths (see core.batching).
        return compute_loss(output, batch, self.loss_fn)

    # --- delegators -------------------------------------------------------
    # The bodies live in sibling modules as functions over this object.  This
    # class stays a plain attribute holder on purpose: tests assign
    # ``trainer.loss_fn`` and ``trainer.scaler`` directly, and proxying state
    # through a session object would silently swallow those assignments.

    def _optimizer_step_block(self) -> tuple[float | None, bool]:
        return step.run_optimizer_step(self)

    def _pipeline_train_step(self, batch: Any) -> StepOutput:
        return step.run_pipeline_step(self, batch)

    def train_step(self, batch: Any) -> StepOutput:
        return step.run_train_step(self, batch)

    def fit(self, data: Any, *, epochs: int = 1, max_steps: int | None = None,
            resume_from: str | os.PathLike[str] | None = None) -> list[StepOutput]:
        return loop.fit(self, data, epochs=epochs, max_steps=max_steps, resume_from=resume_from)

    def evaluate(self, data: Any) -> dict[str, float]:
        return evaluation.evaluate(self, data)

    def _reduce_eval_stats(self, local_sum: float, local_count: float) -> tuple[float, float]:
        return evaluation.reduce_eval_stats(self, local_sum, local_count)

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        checkpointing.save_checkpoint(self, path)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        return checkpointing.load_checkpoint(self, path)

    def close(self) -> None:
        self.overlap.drain(self.config.overlap.drain_timeout_s)
        self.offload.close(self.optimizer)
        self.runtime.close()
