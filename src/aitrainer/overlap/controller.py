"""Frozen dependency plan and unified drain controller."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any, Callable
from ..lifecycle import AsyncOp, AsyncState, ExecutionScheduler, LifecycleError

@dataclass(frozen=True)
class OverlapRecord:
    name: str; submitted_at: float; completed_at: float; wait_seconds: float
    overlapped_seconds: float = 0.0; bytes: int = 0; kind: str = "unknown"

class OverlapController:
    def __init__(self, scheduler: ExecutionScheduler | None = None,
                 max_inflight_ops: int = 2, max_inflight_bytes: int = 0,
                 budgets: Any | None = None) -> None:
        self.scheduler = scheduler or ExecutionScheduler()
        self.max_inflight_ops = max(1, int(max_inflight_ops)); self.max_inflight_bytes = max(0, int(max_inflight_bytes))
        self.records: list[OverlapRecord] = []; self._deps: dict[int, tuple[AsyncOp, ...]] = {}; self._ops: list[AsyncOp] = []
        self._frozen = False; self._plan: tuple[AsyncOp, ...] = (); self._budgets = budgets
        # Real concurrency high-water mark; summary() used to report the total
        # number of recorded operations here instead.
        self._inflight_peak = 0
    def add_dependency(self, op: AsyncOp, *predecessors: AsyncOp) -> None:
        if self._frozen: raise LifecycleError("dependency plan is frozen")
        self._deps[id(op)] = tuple(predecessors); op.dependencies = tuple(predecessors)
        if not any(id(existing) == id(op) for existing in self._ops): self._ops.append(op)
    def freeze(self) -> tuple[AsyncOp, ...]:
        self._plan = tuple(self._ops); self._frozen = True; return self._plan
    @property
    def plan_frozen(self) -> bool: return self._frozen
    def _enforce_budget(self, bytes_: int) -> None:
        # Wait for AND release the oldest work until both budgets fit.  Waiting
        # alone left the op in scheduler.pending forever, so every later call
        # re-waited that same already-WAITED op (instant) and neither the op count
        # nor the byte budget ever blocked again.
        while True:
            pending = self.scheduler.pending
            if not pending:
                return
            over_ops = len(pending) >= self.max_inflight_ops
            over_bytes = bool(self.max_inflight_bytes) and (
                sum(max(0, op.bytes) for op in pending) + bytes_ > self.max_inflight_bytes)
            if not (over_ops or over_bytes):
                return
            oldest = pending[0]
            self.wait(oldest)
            self.release(oldest)
    def submit_op(self, op: AsyncOp) -> AsyncOp:
        self._enforce_budget(int(op.bytes))
        if op.state == AsyncState.CREATED: op.submit()
        registered = self.scheduler.register(op)
        self._inflight_peak = max(self._inflight_peak, len(self.scheduler.pending))
        return registered
    def submit(self, name: str, operation: Callable[[], Any], *, context: dict[str, Any] | None = None) -> Any:
        del context
        # Disabled is the synchronous baseline, not an unmeasured path: the call
        # still runs inline and is still recorded, with zero overlapped time, so
        # summary() reports the baseline honestly instead of reporting nothing.
        started = time.perf_counter(); result = operation(); ended = time.perf_counter()
        self.records.append(OverlapRecord(name, started, ended, 0.0)); return result
    def register(self, op: AsyncOp) -> AsyncOp: return self.submit_op(op)
    def wait(self, op: AsyncOp, timeout: float | None = None) -> Any:
        started = time.perf_counter(); result = op.wait(timeout); ended = time.perf_counter()
        self.records.append(OverlapRecord(op.name, op.submitted_at, ended, ended - started, bytes=op.bytes))
        return result
    def release(self, op: AsyncOp) -> None: op.release()
    def drain(self, timeout: float | None = None) -> None: self.scheduler.drain(timeout)
    def summary(self) -> dict[str, float]:
        total = sum(max(0.0, r.completed_at-r.submitted_at) for r in self.records); wait = sum(r.wait_seconds for r in self.records)
        overlap = sum(r.overlapped_seconds for r in self.records)
        return {"operations": float(len(self.records)), "elapsed_seconds": total, "wait_seconds": wait,
                "overlapped_seconds": overlap, "overlap_ratio": overlap/total if total else 0.0,
                "inflight_peak": float(self._inflight_peak)}
