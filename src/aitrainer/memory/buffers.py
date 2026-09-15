"""Quota-bound CPU buffer pooling.

The pool deliberately owns only CPU allocations.  CUDA streams and events are
owned by the offload backend that performs a transfer; this keeps reuse and
transfer lifetimes explicit and avoids a module-level allocator.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Callable


class BufferPoolError(RuntimeError):
    """Raised when a buffer cannot be safely allocated or returned."""


@dataclass(frozen=True)
class BufferKey:
    shape: tuple[int, ...]
    dtype: Any
    device: str = "cpu"
    layout: str = "contiguous"

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape):
            raise ValueError("buffer shape must contain non-negative integers")
        if not str(self.device).startswith("cpu"):
            raise ValueError("PinnedBufferPool only allocates CPU buffers")

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * _dtype_size(self.dtype)


@dataclass
class _Entry:
    key: BufferKey
    tensor: Any
    nbytes: int
    in_use: bool = True


def _dtype_size(dtype: Any) -> int:
    """Best-effort dtype size without importing torch at module import time."""
    if isinstance(dtype, str):
        name = dtype.lower().replace("torch.", "")
        return {"bool": 1, "uint8": 1, "int8": 1, "float8_e4m3fn": 1,
                "float8_e5m2": 1, "int16": 2, "float16": 2, "bfloat16": 2,
                "int32": 4, "float32": 4, "int64": 8, "float64": 8}.get(name, 4)
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    name = str(dtype).lower()
    for token, size in (("float64", 8), ("int64", 8), ("float32", 4), ("int32", 4),
                        ("float16", 2), ("bfloat16", 2), ("int16", 2), ("bool", 1),
                        ("uint8", 1), ("int8", 1)):
        if token in name:
            return size
    return 4


class PinnedBufferPool:
    """Reusable CPU tensor pool with a hard byte quota.

    ``pin_memory`` is honoured only when CUDA is available.  On a CPU-only
    installation regular CPU tensors are used because pinned allocation is not
    meaningful there; the ``stats()['pinned']`` field makes that state visible.
    """

    def __init__(self, max_bytes: int, *, pin_memory: bool = True,
                 allocator: Callable[[BufferKey, bool], Any] | None = None) -> None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.pin_memory = bool(pin_memory)
        self._allocator = allocator
        self._free: dict[BufferKey, list[_Entry]] = defaultdict(list)
        self._entries: dict[int, _Entry] = {}
        self._allocated_bytes = 0
        self._external_bytes = 0
        self._in_use_bytes = 0
        self._hits = 0
        self._misses = 0
        self._pinned = False

    def _allocate(self, key: BufferKey) -> Any:
        if self._allocator is not None:
            return self._allocator(key, self.pin_memory)
        try:
            import torch
        except ImportError as exc:
            raise BufferPoolError("PyTorch is required to allocate offload buffers") from exc
        dtype = key.dtype
        if isinstance(dtype, str):
            name = dtype.replace("torch.", "")
            if not hasattr(torch, name):
                raise BufferPoolError(f"unsupported buffer dtype {dtype!r}")
            dtype = getattr(torch, name)
        if key.layout == "strided":
            # ``empty_strided(shape, shape)`` passed the SHAPE as the strides: the
            # buffer was over-allocated, non-contiguous and aliased itself, while
            # the quota was charged numel*itemsize.  BufferKey carries no strides,
            # so this layout cannot be honoured -- refuse it.
            raise BufferPoolError(
                "layout='strided' is not supported: BufferKey carries no strides; "
                "use layout='contiguous' or pass an explicit allocator")
        want_pin = self.pin_memory and bool(getattr(torch.cuda, "is_available", lambda: False)())
        tensor = torch.empty(key.shape, device="cpu", dtype=dtype, pin_memory=want_pin)
        self._pinned = self._pinned or bool(getattr(tensor, "is_pinned", lambda: False)())
        return tensor

    def acquire(self, key: BufferKey) -> Any:
        if not isinstance(key, BufferKey):
            raise TypeError("acquire expects a BufferKey")
        available = self._free.get(key)
        if available:
            entry = available.pop()
            entry.in_use = True
            self._entries[id(entry.tensor)] = entry
            self._in_use_bytes += entry.nbytes
            self._hits += 1
            return entry.tensor
        if self._allocated_bytes + self._external_bytes + key.nbytes > self.max_bytes:
            raise BufferPoolError(
                f"CPU offload buffer quota exceeded: requested={key.nbytes} "
                f"allocated={self._allocated_bytes} quota={self.max_bytes}"
            )
        tensor = self._allocate(key)
        # Charge what was ACTUALLY allocated.  Clamping down to the requested size
        # made an oversized allocation (the old strided path allocated ~1000x the
        # quota) look free to the next admission check.
        actual = int(getattr(tensor, "numel", lambda: key.numel)()) * _dtype_size(getattr(tensor, "dtype", key.dtype))
        entry = _Entry(key, tensor, actual)
        self._entries[id(tensor)] = entry
        self._allocated_bytes += actual
        self._in_use_bytes += actual
        self._misses += 1
        return tensor

    def allocate(self, key: BufferKey) -> Any:
        """Alias expressing ownership transfer at call sites that allocate."""
        return self.acquire(key)

    def release(self, tensor: Any) -> None:
        entry = self._entries.get(id(tensor))
        if entry is None:
            raise BufferPoolError("attempted to release a tensor not owned by this pool")
        if not entry.in_use:
            raise BufferPoolError("buffer was released more than once")
        entry.in_use = False
        self._in_use_bytes -= entry.nbytes
        self._free[entry.key].append(entry)

    def reserve_external(self, nbytes: int) -> None:
        """Reserve quota for a layout-specific allocation owned by a backend."""
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("nbytes must be a non-negative integer")
        if self._allocated_bytes + self._external_bytes + nbytes > self.max_bytes:
            raise BufferPoolError(
                f"CPU offload buffer quota exceeded: requested={nbytes} "
                f"allocated={self._allocated_bytes + self._external_bytes} quota={self.max_bytes}"
            )
        self._external_bytes += nbytes

    def release_external(self, nbytes: int) -> None:
        if nbytes < 0 or nbytes > self._external_bytes:
            raise BufferPoolError("invalid external buffer reservation release")
        self._external_bytes -= nbytes

    def clear(self) -> None:
        if any(entry.in_use for entry in self._entries.values()):
            raise BufferPoolError("cannot clear a pool with buffers still in use")
        if self._external_bytes:
            raise BufferPoolError("cannot clear a pool with external reservations")
        self._free.clear()
        self._entries.clear()
        self._allocated_bytes = 0
        self._external_bytes = 0
        self._in_use_bytes = 0

    def stats(self) -> dict[str, int | bool]:
        return {"allocated_bytes": self._allocated_bytes, "external_bytes": self._external_bytes,
                "in_use_bytes": self._in_use_bytes,
                "free_bytes": self._allocated_bytes - self._in_use_bytes,
                "quota_bytes": self.max_bytes, "in_flight": sum(e.in_use for e in self._entries.values()),
                "hits": self._hits, "misses": self._misses, "pinned": self._pinned}
