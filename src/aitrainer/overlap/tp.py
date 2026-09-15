"""TP bulk overlap orchestration; tensor math remains in parallel.tp."""
from __future__ import annotations
from typing import Any, Callable
from ..lifecycle import AsyncOp, ExecutionScheduler

class TPBulkOverlap:
    def __init__(self, *, enabled: bool = False, scheduler: ExecutionScheduler | None = None) -> None: self.enabled = bool(enabled); self.scheduler = scheduler or ExecutionScheduler()
    def collective(self, name: str, fn: Callable[[], Any], *, bytes_: int = 0) -> AsyncOp:
        if not self.enabled: fn(); op = AsyncOp(name, bytes=bytes_).submit(); op.mark_ready(); op.wait(); return op
        op = AsyncOp(name, handle=fn, bytes=bytes_, _wait_fn=lambda f: f()).submit(); return self.scheduler.register(op)
    def drain(self) -> None: self.scheduler.drain()
