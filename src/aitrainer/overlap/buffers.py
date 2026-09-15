"""Generation checked ping-pong leases for temporary communication buffers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.lifecycle import AsyncOp, AsyncState, LifecycleError


@dataclass
class BufferLease:
    buffer: Any; index: int; generation: int; owner: str; released: bool = False
    def release(self) -> None: self.released = True

class PingPongBuffer:
    def __init__(self, buffers: tuple[Any, Any] | list[Any], *, name: str = "buffer") -> None:
        if len(buffers) != 2: raise ValueError("ping-pong buffer requires exactly two buffers")
        self.buffers = tuple(buffers); self.name = name; self._generation = [0, 0]; self._busy: list[AsyncOp | None] = [None, None]; self._next = 0
    def acquire(self, owner: str, *, generation: int | None = None) -> BufferLease:
        for offset in range(2):
            idx = (self._next+offset)%2; op = self._busy[idx]
            if op is not None:
                if op.state not in {AsyncState.READY, AsyncState.WAITED, AsyncState.RELEASED}: continue
                op.release(); self._busy[idx] = None
            self._next = (idx+1)%2; self._generation[idx] += 1
            gen = self._generation[idx]
            if generation is not None and generation != gen: raise LifecycleError(f"{self.name} generation mismatch: expected {generation}, actual {gen}")
            return BufferLease(self.buffers[idx], idx, gen, owner)
        raise LifecycleError(f"{self.name} has no recyclable buffer")
    def attach(self, lease: BufferLease, op: AsyncOp) -> None:
        if lease.released or lease.generation != self._generation[lease.index]: raise LifecycleError("stale buffer lease")
        self._busy[lease.index] = op
    def release(self, lease: BufferLease, *, generation: int | None = None) -> None:
        if lease.released: return
        if generation is not None and generation != lease.generation: raise LifecycleError("buffer generation mismatch")
        op = self._busy[lease.index]
        if op is not None and op.state not in {AsyncState.READY, AsyncState.WAITED, AsyncState.RELEASED}: raise LifecycleError("buffer is still in use")
        lease.release(); self._busy[lease.index] = None
