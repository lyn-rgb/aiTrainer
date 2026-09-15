"""Unified bounded H2D/D2H transfer scheduler with synchronous fallback."""
from __future__ import annotations
from typing import Any, Callable
from ..lifecycle import AsyncOp, ExecutionScheduler
from .backpressure import Backpressure, ResourceBudget

class TransferScheduler:
    def __init__(self, *, enabled: bool = False, max_transfer_bytes: int = 0, scheduler: ExecutionScheduler | None = None) -> None:
        self.enabled = bool(enabled); self.scheduler = scheduler or ExecutionScheduler(); self.backpressure = Backpressure(ResourceBudget(max_transfer_bytes=max_transfer_bytes or 0x7FFFFFFF)); self.records: list[dict[str, Any]] = []
    def submit(self, name: str, operation: Callable[[], Any], *, bytes_: int = 0, direction: str = "h2d") -> AsyncOp:
        if direction not in {"h2d", "d2h"}: raise ValueError("direction must be h2d or d2h")
        if not self.enabled: operation(); op = AsyncOp(name, bytes=bytes_).submit(); op.mark_ready(); op.wait(); op.release(); return op
        op = AsyncOp(name, handle=operation, bytes=bytes_, _wait_fn=lambda fn: fn()).submit(); self.backpressure.admit(op); self.scheduler.register(op); self.records.append({"name": name, "direction": direction, "bytes": bytes_}); return op
    def drain(self, timeout: float | None = None) -> None: self.scheduler.drain(timeout); self.backpressure.drain()
