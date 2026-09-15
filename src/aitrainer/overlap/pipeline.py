"""Stable-key PP operation queues used by GPipe and 1F1B schedules."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..lifecycle import AsyncOp, ExecutionScheduler, LifecycleError

@dataclass(frozen=True)
class PipelineKey:
    stage: int; virtual_stage: int; micro_batch: int; direction: str

class PipelineHandleQueue:
    def __init__(self, scheduler: ExecutionScheduler | None = None) -> None: self.scheduler = scheduler or ExecutionScheduler(); self._handles: dict[PipelineKey, AsyncOp] = {}
    def post(self, key: PipelineKey, handle: Any, *, bytes_: int = 0) -> AsyncOp:
        if key in self._handles: raise LifecycleError(f"duplicate pipeline handle {key}")
        op = AsyncOp(f"p2p:{key.direction}:{key.micro_batch}", handle=handle, bytes=bytes_).submit(); self._handles[key] = self.scheduler.register(op); return op
    def wait(self, key: PipelineKey, timeout: float | None = None) -> Any:
        op = self._handles.pop(key, None)
        if op is None: raise LifecycleError(f"missing pipeline handle {key}")
        result = op.wait(timeout); op.release(); return result
    def drain(self, timeout: float | None = None) -> None: self.scheduler.drain(timeout); self._handles.clear()
