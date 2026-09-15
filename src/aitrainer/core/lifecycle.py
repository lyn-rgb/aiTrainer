"""Explicit lifecycle primitives shared by communication and offload paths."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable

class LifecycleError(RuntimeError):
    """Raised for invalid operation state transitions or failed cleanup."""

class AsyncState(str, Enum):
    CREATED = "created"; SUBMITTED = "submitted"; READY = "ready"; WAITED = "waited"
    RELEASED = "released"; FAILED = "failed"; TIMED_OUT = "timed_out"

_TRANSITIONS = {
    AsyncState.CREATED: {AsyncState.SUBMITTED, AsyncState.FAILED},
    AsyncState.SUBMITTED: {AsyncState.READY, AsyncState.FAILED, AsyncState.TIMED_OUT},
    AsyncState.READY: {AsyncState.WAITED, AsyncState.RELEASED, AsyncState.FAILED},
    AsyncState.WAITED: {AsyncState.RELEASED, AsyncState.FAILED},
    AsyncState.RELEASED: set(), AsyncState.FAILED: {AsyncState.RELEASED},
    AsyncState.TIMED_OUT: {AsyncState.RELEASED, AsyncState.FAILED},
}

@dataclass
class AsyncOp:
    """A stateful, idempotently waitable asynchronous operation."""
    name: str
    handle: Any = None
    group: Any = None
    tensor: Any = None
    rank: int | None = None
    micro_batch: int | None = None
    bucket: str | None = None
    bytes: int = 0
    producer_stream: Any = None
    producer_event: Any = None
    consumer_stream: Any = None
    consumer_event: Any = None
    buffer_lease: Any = None
    generation: int | None = None
    dependencies: tuple["AsyncOp", ...] = ()
    submitted_at: float = 0.0
    _wait_fn: Callable[[Any], Any] | None = None
    _release_fn: Callable[[], Any] | None = None
    _state: AsyncState = field(default=AsyncState.CREATED, init=False)
    _result: Any = field(default=None, init=False)
    _error: BaseException | None = field(default=None, init=False)
    _completed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.submitted_at: self.submitted_at = time.monotonic()
    @property
    def state(self) -> AsyncState: return self._state
    @property
    def completed(self) -> bool: return self._state in {AsyncState.READY, AsyncState.WAITED, AsyncState.RELEASED}
    def _transition(self, state: AsyncState) -> None:
        if state not in _TRANSITIONS[self._state]:
            raise LifecycleError(f"illegal async op transition {self.name!r}: {self._state.value}->{state.value}; rank={self.rank} group={self.group!r} generation={self.generation}")
        self._state = state
    def submit(self) -> "AsyncOp":
        if self._state == AsyncState.CREATED:
            self.submitted_at = time.monotonic(); self._transition(AsyncState.SUBMITTED)
        elif self._state != AsyncState.SUBMITTED:
            raise LifecycleError(f"cannot submit op {self.name!r} from {self._state.value}")
        return self
    def mark_ready(self, result: Any = None) -> Any:
        if self._state == AsyncState.READY: return self._result
        if self._state != AsyncState.SUBMITTED: raise LifecycleError(f"cannot mark_ready op {self.name!r} from {self._state.value}")
        self._result = result; self._completed = True; self._transition(AsyncState.READY); return result
    def fail(self, error: BaseException) -> None:
        if self._state in {AsyncState.RELEASED, AsyncState.FAILED}: return
        self._error = error; self._state = AsyncState.FAILED
    def wait(self, timeout: float | None = None) -> Any:
        if self._state == AsyncState.WAITED: return self._result
        if self._state == AsyncState.RELEASED:
            if not self._completed:
                # release() is legal from TIMED_OUT/FAILED, and returning
                # self._result (None) here turned a failed operation into a
                # silent empty success for anything that re-waited.
                raise LifecycleError(
                    f"async op {self.name!r} was released without completing; "
                    "its result was never produced")
            return self._result
        if self._state == AsyncState.FAILED: raise LifecycleError(f"async op {self.name!r} failed") from self._error
        if self._state == AsyncState.TIMED_OUT: raise TimeoutError(f"async op {self.name!r} previously timed out")
        if self._state == AsyncState.CREATED: self.submit()
        for dependency in self.dependencies: dependency.wait(timeout)
        if self._state == AsyncState.READY:
            self._transition(AsyncState.WAITED)
            return self._result
        if timeout is not None and time.monotonic() - self.submitted_at > timeout:
            self._state = AsyncState.TIMED_OUT
            raise TimeoutError(f"async op {self.name!r} exceeded timeout; rank={self.rank} micro_batch={self.micro_batch}")
        try:
            result = self._wait_fn(self.handle) if self._wait_fn is not None else (self.handle.wait() if hasattr(self.handle, "wait") else None)
            self.mark_ready(result); self._transition(AsyncState.WAITED); return result
        except BaseException as exc:
            self.fail(exc); raise
    def release(self) -> None:
        if self._state == AsyncState.RELEASED: return
        if self._state in {AsyncState.CREATED, AsyncState.SUBMITTED}: self.wait()
        try:
            if self._release_fn is not None: self._release_fn()
        except BaseException as exc:
            self.fail(exc); raise
        if self._state != AsyncState.RELEASED: self._transition(AsyncState.RELEASED)

class ExecutionScheduler:
    """Own pending operations and drain them in reverse submission order."""
    def __init__(self) -> None: self._pending: list[AsyncOp] = []
    def register(self, op: AsyncOp) -> AsyncOp:
        if not any(id(existing) == id(op) for existing in self._pending) and op.state != AsyncState.RELEASED: self._pending.append(op)
        return op
    def submit(self, name: str, handle: Any = None, **context: Any) -> AsyncOp:
        return self.register(AsyncOp(name=name, handle=handle, **context).submit())
    def drain(self, timeout: float | None = None) -> None:
        errors: list[BaseException] = []
        for op in reversed(self._pending):
            try: op.wait(timeout)
            except BaseException as exc: errors.append(exc)
            finally:
                try: op.release()
                except BaseException as exc: errors.append(exc)
        self._pending.clear()
        if errors: raise LifecycleError(f"failed to drain {len(errors)} async operations") from errors[0]
    @property
    def pending(self) -> tuple[AsyncOp, ...]: return tuple(op for op in self._pending if op.state != AsyncState.RELEASED)

def drain(scheduler: ExecutionScheduler | None = None) -> None:
    if scheduler is not None: scheduler.drain()

__all__ = ["AsyncOp", "AsyncState", "ExecutionScheduler", "LifecycleError", "drain"]
