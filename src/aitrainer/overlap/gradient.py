"""Backward-ready gradient buckets with accumulation and sync fallback."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.lifecycle import AsyncOp, ExecutionScheduler


@dataclass
class GradientBucket:
    name: str; max_bytes: int; on_flush: Callable[[tuple[Any, ...]], Any] | None = None
    tensors: list[Any] = field(default_factory=list); bytes: int = 0; ready: bool = False; flush_count: int = 0
    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.max_bytes, int) or self.max_bytes < 1: raise ValueError("bucket name and positive max_bytes are required")
    @staticmethod
    def _nbytes(t: Any) -> int:
        return int(t.numel())*int(t.element_size()) if hasattr(t, "numel") and hasattr(t, "element_size") else max(1, len(t) if hasattr(t, "__len__") else 1)
    def add(self, tensor: Any, value: Any | None = None) -> bool:
        tensor = value if value is not None else tensor
        if self.ready: raise RuntimeError(f"gradient bucket {self.name!r} is ready; flush before reusing")
        size = self._nbytes(tensor)
        if self.tensors and self.bytes + size > self.max_bytes: self.ready = True; return False
        self.tensors.append(tensor); self.bytes += size; self.ready = self.bytes >= self.max_bytes; return True
    register = add
    @property
    def is_ready(self) -> bool:
        return self.ready
    def mark_ready(self) -> None: self.ready = True
    def flush(self, force: bool = True) -> Any:
        if not self.tensors: self.ready = False; return None
        values = tuple(self.tensors); result = self.on_flush(values) if self.on_flush else values
        self.tensors.clear(); self.bytes = 0; self.ready = False; self.flush_count += 1; return result

class GradientBucketReducer:
    def __init__(self, bucket_bytes: int, *, accumulation_steps: int = 1, reduce_fn: Callable[[tuple[Any, ...]], Any] | None = None, scheduler: ExecutionScheduler | None = None, fsdp: bool = False) -> None:
        if fsdp: raise ValueError("gradient bucket reducer cannot be combined with FSDP reducer")
        if accumulation_steps < 1: raise ValueError("accumulation_steps must be positive")
        self.bucket_bytes = bucket_bytes; self.accumulation_steps = accumulation_steps; self.reduce_fn = reduce_fn; self.scheduler = scheduler or ExecutionScheduler(); self.buckets: list[GradientBucket] = []; self.direct_reductions: list[Any] = []; self.ready_order: list[str] = []; self._microbatch = 0; self._finished = False
    def register(self, name: str, tensor: Any) -> None:
        # New gradient work opens a fresh accumulation window.  Without this the
        # one-shot `_finished` latch let finish_grad_sync() reduce exactly once
        # per process lifetime and then silently skip every later window.
        self._finished = False
        if GradientBucket._nbytes(tensor) > self.bucket_bytes:
            self.direct_reductions.append((name, tensor)); self.ready_order.append(name); return
        bucket = self.buckets[-1] if self.buckets and not self.buckets[-1].ready else None
        if bucket is None or not bucket.add(tensor):
            bucket = GradientBucket(name, self.bucket_bytes, None)
            bucket.add(tensor)
            self.buckets.append(bucket)
        self.ready_order.append(name)
    def mark_microbatch_end(self) -> None: self._microbatch += 1
    def finish_grad_sync(self) -> None:
        if self._finished: return
        if self._microbatch and self._microbatch % self.accumulation_steps != 0: return
        if self.reduce_fn is None:
            # Flushing without a reducer would clear the buckets and drop every
            # registered gradient with no error.  Refuse instead.
            raise ValueError("GradientBucketReducer requires reduce_fn to flush gradients")
        for bucket in self.buckets:
            if bucket.tensors:
                values = tuple(bucket.tensors)
                if self.reduce_fn is not None:
                    result = self.reduce_fn(values)
                    if isinstance(result, AsyncOp): self.scheduler.register(result)
            bucket.flush()
        for _, tensor in self.direct_reductions:
            if self.reduce_fn is not None:
                result = self.reduce_fn((tensor,))
                if isinstance(result, AsyncOp): self.scheduler.register(result)
        self.direct_reductions.clear()
        self._finished = True
    def reset(self) -> None: self._finished = False; self._microbatch = 0
