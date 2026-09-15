"""Composition root for opt-in CPU offload backends."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..config import OffloadConfig
from ..lifecycle import ExecutionScheduler
from ..memory import PinnedBufferPool
from ..overlap.transfer import TransferScheduler
from .activation import ActivationOffloader
from .optimizer import CPUOptimizerStateOffloader
from .parameter import ParameterOffloader


class OffloadError(RuntimeError):
    """Raised for unsupported or invalid offload operations."""


class OffloadManager:
    """Own all enabled offload resources and expose explicit lifecycle calls."""

    def __init__(self, config: OffloadConfig | None = None, *, scheduler: ExecutionScheduler | None = None,
                 overlap_config: Any | None = None) -> None:
        self.config = config or OffloadConfig()
        self.config.validate()
        self.scheduler = scheduler or ExecutionScheduler()
        transfer_enabled = bool(getattr(overlap_config, "enable_transfer_overlap", False))
        transfer_budget = int(getattr(overlap_config, "max_transfer_bytes", 0) or self.config.max_cpu_bytes)
        self.transfer = TransferScheduler(enabled=transfer_enabled, scheduler=self.scheduler,
                                          max_transfer_bytes=transfer_budget)
        self.pool = PinnedBufferPool(self.config.max_cpu_bytes, pin_memory=self.config.pin_memory)
        self.enabled = bool(self.config.enabled)
        self.optimizer: CPUOptimizerStateOffloader | None = None
        self.parameter: ParameterOffloader | None = None
        self.activation: ActivationOffloader | None = None
        if self.enabled:
            if self.config.optimizer:
                self.optimizer = CPUOptimizerStateOffloader(pin_memory=self.config.pin_memory, pool=self.pool)
            if self.config.parameter:
                self.parameter = ParameterOffloader(pin_memory=self.config.pin_memory, scheduler=self.scheduler,
                                                    pool=self.pool)
                if overlap_config is not None:
                    self.parameter.trace.max_prefetched_bytes = int(getattr(overlap_config, "parameter_prefetch_bytes", 0))
            if self.config.activation:
                self.activation = ActivationOffloader(threshold_bytes=self.config.activation_threshold_bytes,
                                                      keep_last=self.config.keep_last_activations,
                                                      pin_memory=self.config.pin_memory, pool=self.pool)

    def register_model(self, model: Any) -> None:
        if self.parameter is not None:
            self.parameter.register_module(model)
        if self.activation is not None:
            self.activation.register_module(model)

    def register_optimizer(self, optimizer: Any) -> None:
        if self.optimizer is not None:
            self.optimizer.register(optimizer)

    def step_optimizer(self, optimizer: Any, step_fn: Any, *, optimizer_dtype: Any | None = None) -> Any:
        if optimizer_dtype is not None:
            from ..precision import cast_optimizer_state
            cast_optimizer_state(optimizer, optimizer_dtype)
        if self.optimizer is None:
            return step_fn()
        self.optimizer.to_device(optimizer)
        try:
            return step_fn()
        finally:
            self.optimizer.to_cpu(optimizer)

    def fetch(self, module: Any, *, device: Any | None = None) -> int:
        return self.parameter.fetch(module=module, device=device) if self.parameter is not None else 0

    def prefetch(self, module: Any, *, device: Any | None = None) -> int:
        return self.parameter.prefetch(module, device=device) if self.parameter is not None else 0

    def prefetch_async(self, module: Any, *, device: Any | None = None) -> Any:
        return self.parameter.prefetch_async(module, device=device) if self.parameter is not None else None

    def release(self, module: Any) -> int:
        return self.parameter.release(module=module) if self.parameter is not None else 0

    def activation_context(self, device: Any | None = None) -> Any:
        return self.activation.context(device) if self.activation is not None else nullcontext()

    def drain(self, optimizer: Any | None = None, *, timeout: float | None = None) -> None:
        if self.optimizer is not None and optimizer is not None:
            self.optimizer.to_cpu(optimizer)
        self.scheduler.drain(timeout)
        self.transfer.drain(timeout)
        if self.activation is not None:
            self.activation.drain()

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {"enabled": self.enabled, "pending_async": len(self.scheduler.pending),
                                  "pool": self.pool.stats()}
        if self.parameter is not None: result["parameter"] = self.parameter.stats()
        if self.optimizer is not None:
            result["optimizer"] = (self.optimizer.state_stats() if self.optimizer.registered
                                    else {"tensors": 0, "elements": 0, "devices": (), "on_cpu": True})
        if self.activation is not None: result["activation"] = self.activation.stats()
        return result

    def close(self, optimizer: Any | None = None) -> None:
        self.drain(optimizer)
        if self.parameter is not None:
            self.parameter.release_all()
            self.parameter.clear()
        if self.optimizer is not None:
            self.optimizer.clear()
        self.activation = None
        self.optimizer = None
        self.parameter = None
        self.pool.clear()
