"""Trace-driven parameter prefetch coordinator with safe invalidation fallback."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.lifecycle import AsyncOp, ExecutionScheduler


@dataclass(frozen=True)
class TraceFingerprint:
    value: str

@dataclass
class TraceEntry:
    module: str; parameters: tuple[str, ...]; step: int; phase: str = "forward"; recompute: bool = False

class ParameterPrefetchCoordinator:
    def __init__(self, *, max_prefetched_bytes: int = 0, scheduler: ExecutionScheduler | None = None, fetch_fn: Callable[[str], Any] | None = None) -> None:
        self.max_prefetched_bytes = max(0, max_prefetched_bytes); self.scheduler = scheduler or ExecutionScheduler(); self.fetch_fn = fetch_fn; self.trace: list[TraceEntry] = []; self._fingerprint: TraceFingerprint | None = None; self.valid = False; self._inflight: dict[str, AsyncOp] = {}; self._prefetched_bytes = 0; self.hits = self.misses = self.late = self.evicted = 0
    def record(self, module: str, parameters: list[str] | tuple[str, ...], *, step: int = 0, phase: str = "forward", recompute: bool = False) -> None: self.trace.append(TraceEntry(module, tuple(parameters), step, phase, recompute))
    def fingerprint(self, config: Any = None) -> TraceFingerprint:
        payload = {"trace": [e.__dict__ for e in self.trace], "config": repr(config)}; value = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(); self._fingerprint = TraceFingerprint(value); return self._fingerprint
    def finalize(self, expected: TraceFingerprint | str | None = None) -> bool:
        fp = self.fingerprint(); self.valid = expected is None or fp.value == (expected.value if isinstance(expected, TraceFingerprint) else expected); return self.valid
    def invalidate(self) -> None:
        self.valid = False
        for name in tuple(self._inflight): self.release(name)
    def fetch(self, name: str, *, bytes_: int = 0) -> AsyncOp | None:
        if not self.valid: self.misses += 1; return None
        if name in self._inflight: self.hits += 1; return self._inflight[name]
        if self.max_prefetched_bytes and self._prefetched_bytes + bytes_ > self.max_prefetched_bytes:
            oldest = next(iter(self._inflight), None)
            if oldest is not None: self.release(oldest); self.evicted += 1
            if self._prefetched_bytes + bytes_ > self.max_prefetched_bytes:
                self.late += 1; return None
        result = self.fetch_fn(name) if self.fetch_fn else None
        wait_fn = (lambda value: value() if callable(value) else value)
        op = AsyncOp(f"param_fetch:{name}", handle=result, bytes=bytes_, _wait_fn=wait_fn).submit()
        self.scheduler.register(op); self._inflight[name] = op; self._prefetched_bytes += max(0, bytes_); return op
    def wait(self, name: str, timeout: float | None = None) -> Any:
        op = self._inflight.get(name); self.misses += op is None
        return op.wait(timeout) if op else (self.fetch_fn(name) if self.fetch_fn else None)
    def release(self, name: str) -> None:
        op = self._inflight.pop(name, None)
        if op is not None:
            op.release(); self._prefetched_bytes = max(0, self._prefetched_bytes - max(0, op.bytes))
