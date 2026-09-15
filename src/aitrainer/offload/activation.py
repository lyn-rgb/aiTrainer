"""Saved-tensor activation offload using PyTorch's local hook context."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ..memory import BufferKey, PinnedBufferPool


class ActivationOffloadError(RuntimeError):
    """Raised when activation hooks cannot preserve tensor semantics."""


@dataclass
class _SavedTensor:
    tensor: Any
    cpu: Any = None
    device: Any = None
    dtype: Any = None
    shape: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()
    pooled: bool = False
    external_reserved: int = 0
    event: Any = None


class ActivationOffloader:
    """Offload eligible saved tensors within an explicit context manager."""

    def __init__(self, *, threshold_bytes: int = 1 << 20, keep_last: int = 1,
                 pin_memory: bool = True, pool: PinnedBufferPool | None = None) -> None:
        if threshold_bytes < 0 or keep_last < 0:
            raise ValueError("activation threshold and keep_last must be non-negative")
        self.threshold_bytes = threshold_bytes
        self.keep_last = keep_last
        self.pin_memory = bool(pin_memory)
        self.pool = pool
        self._parameter_ids: set[int] = set()
        self._buffer_ids: set[int] = set()
        self._records: deque[_SavedTensor] = deque()
        self._offloaded = 0
        self._restored = 0

    def register_module(self, module: Any) -> None:
        if hasattr(module, "parameters"):
            self._parameter_ids.update(id(item) for item in module.parameters())
        if hasattr(module, "buffers"):
            self._buffer_ids.update(id(item) for item in module.buffers())

    def _eligible(self, tensor: Any) -> bool:
        if not hasattr(tensor, "numel") or not hasattr(tensor, "device"):
            return False
        if str(tensor.device) == "cpu" or id(tensor) in self._parameter_ids or id(tensor) in self._buffer_ids:
            return False
        if int(tensor.numel()) * int(tensor.element_size()) <= self.threshold_bytes:
            return False
        if getattr(tensor, "_base", None) is not None:
            return False
        if any(int(s) == 0 for s in getattr(tensor, "stride", lambda: ())()):
            return False
        return True

    def _offload(self, record: _SavedTensor) -> None:
        tensor = record.tensor
        try:
            import torch
            shape = tuple(int(x) for x in tensor.shape)
            stride = tuple(int(x) for x in tensor.stride())
            want_pin = self.pin_memory and bool(getattr(torch.cuda, "is_available", lambda: False)())
            contiguous = bool(getattr(tensor, "is_contiguous", lambda: False)())
            if self.pool is not None and contiguous:
                cpu = self.pool.acquire(BufferKey(shape, tensor.dtype))
                record.pooled = True
            else:
                if self.pool is not None:
                    record.external_reserved = int(tensor.numel()) * int(tensor.element_size())
                    self.pool.reserve_external(record.external_reserved)
                cpu = torch.empty_strided(shape, stride, dtype=tensor.dtype, device="cpu", pin_memory=want_pin)
            cpu.copy_(tensor.detach(), non_blocking=want_pin)
            if want_pin and str(tensor.device).startswith("cuda"):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(tensor.device))
                record.event = event
        except ImportError as exc:
            raise ActivationOffloadError("PyTorch is required for activation offload") from exc
        except Exception as exc:
            if self.pool is not None and record.external_reserved:
                self.pool.release_external(record.external_reserved)
                record.external_reserved = 0
            if self.pool is not None and record.pooled and "cpu" in locals():
                self.pool.release(cpu)
                record.pooled = False
            raise ActivationOffloadError(f"failed to stage activation on CPU: {exc}") from exc
        record.cpu = cpu
        record.tensor = None
        self._offloaded += 1

    def _pack(self, tensor: Any) -> Any:
        record = _SavedTensor(tensor=tensor, device=tensor.device, dtype=tensor.dtype,
                              shape=tuple(tensor.shape), stride=tuple(tensor.stride()))
        self._records.append(record)
        if self._eligible(tensor) and len(self._records) > self.keep_last:
            candidate = self._records[-self.keep_last - 1]
            if candidate.tensor is not None and self._eligible(candidate.tensor):
                self._offload(candidate)
        return record

    def _unpack(self, record: _SavedTensor) -> Any:
        if record.cpu is None:
            if record.tensor is None:
                # drain() ran before this backward finished.  Returning None here
                # let autograd consume a missing activation as though it were
                # real, producing silently wrong gradients.
                raise ActivationOffloadError(
                    "activation was drained before its backward ran; the graph "
                    "outlived the offload context")
            return record.tensor
        if record.event is not None:
            record.event.synchronize()
        # The stable backend uses a synchronous H2D copy before returning the
        # storage.  The CPU copy is deliberately NOT released here: autograd may
        # call unpack more than once for one saved tensor (retain_graph, double
        # backward, a custom Function reading saved_tensors twice) and every call
        # must observe the same data.  drain() returns it when the context exits.
        restored = record.cpu.to(device=record.device, non_blocking=False)
        if tuple(getattr(restored, "shape", ())) != record.shape:
            raise ActivationOffloadError("restored activation shape does not match original")
        self._restored += 1
        return restored

    @contextmanager
    def context(self, device: Any | None = None) -> Iterator[None]:
        del device  # device is retained in each saved record and need not be global.
        try:
            import torch
        except ImportError as exc:
            raise ActivationOffloadError("PyTorch is required for activation offload") from exc
        hooks = getattr(getattr(torch, "autograd", None), "graph", None)
        if hooks is None or not hasattr(hooks, "saved_tensors_hooks"):
            raise ActivationOffloadError("saved_tensors_hooks is unavailable in this PyTorch version")
        self._records.clear()
        try:
            with hooks.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self.drain()

    def stats(self) -> dict[str, int]:
        return {"offloaded": self._offloaded, "restored": self._restored,
                "active_records": len(self._records)}

    def drain(self) -> None:
        for record in self._records:
            if self.pool is not None and record.pooled and record.cpu is not None:
                self.pool.release(record.cpu)
            if self.pool is not None and record.external_reserved:
                self.pool.release_external(record.external_reserved)
            # Clear the handles so a graph that outlives this context fails loudly
            # in _unpack instead of reading storage the pool may already have
            # handed to another owner (a second release could even be accepted,
            # letting two live consumers share one buffer).
            record.cpu = None
            record.external_reserved = 0
            record.pooled = False
        self._records.clear()
