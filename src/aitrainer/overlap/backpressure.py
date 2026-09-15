"""Bounded in-flight resource accounting."""
from __future__ import annotations
from dataclasses import dataclass
from ..lifecycle import AsyncOp, AsyncState, LifecycleError

@dataclass(frozen=True)
class ResourceBudget:
    max_inflight_ops: int = 2; max_inflight_bytes: int = 0; max_prefetched_param_bytes: int = 0; max_transfer_bytes: int = 0
    def validate(self) -> None:
        if self.max_inflight_ops < 1 or min(self.max_inflight_bytes, self.max_prefetched_param_bytes, self.max_transfer_bytes) < 0: raise ValueError("invalid resource budget")

class Backpressure:
    def __init__(self, budget: ResourceBudget = ResourceBudget()) -> None:
        budget.validate(); self.budget = budget; self._ops: list[AsyncOp] = []
    @property
    def inflight_bytes(self) -> int: return sum(max(0, int(o.bytes)) for o in self._ops if o.state not in {AsyncState.RELEASED, AsyncState.WAITED, AsyncState.READY})
    def admit(self, op: AsyncOp) -> None:
        if len(self._ops) >= self.budget.max_inflight_ops or (self.budget.max_inflight_bytes and self.inflight_bytes + op.bytes > self.budget.max_inflight_bytes):
            self.drain_oldest()
        if len(self._ops) >= self.budget.max_inflight_ops or (self.budget.max_inflight_bytes and self.inflight_bytes + op.bytes > self.budget.max_inflight_bytes): raise LifecycleError("in-flight resource budget exceeded")
        self._ops.append(op)
    def drain_oldest(self) -> None:
        if self._ops:
            op = self._ops.pop(0); op.wait(); op.release()
    def drain(self) -> None:
        while self._ops: self.drain_oldest()
