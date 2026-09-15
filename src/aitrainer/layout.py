"""Tensor layout contracts used by synchronous parallel backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal


LayoutKind = Literal["replicated", "partial", "sequence", "head", "hidden"]


class LayoutError(ValueError):
    """Raised when a tensor and its declared layout disagree."""


@dataclass(frozen=True)
class TensorLayout:
    kind: LayoutKind
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    shard_dim: int | None = None
    group: Any = None
    requires_grad: bool = False
    valid_lengths: Any = None

    def __post_init__(self) -> None:
        if self.kind not in {"replicated", "partial", "sequence", "head", "hidden"}:
            raise LayoutError(f"unknown layout kind {self.kind!r}")
        if any(not isinstance(size, int) or size < 0 for size in self.shape):
            raise LayoutError(f"invalid layout shape {self.shape!r}")
        if self.kind in {"replicated", "partial"} and self.shard_dim is not None:
            raise LayoutError(f"{self.kind} layout cannot declare shard_dim")
        if self.kind in {"sequence", "head", "hidden"}:
            if self.shard_dim is None or not -len(self.shape) <= self.shard_dim < len(self.shape):
                raise LayoutError(f"{self.kind} layout requires a valid shard_dim")

    @classmethod
    def from_tensor(cls, tensor: Any, kind: LayoutKind = "replicated", **kwargs: Any) -> "TensorLayout":
        return cls(kind=kind, shape=tuple(tensor.shape), dtype=tensor.dtype,
                   device=tensor.device, requires_grad=bool(tensor.requires_grad), **kwargs)

    def validate_tensor(self, tensor: Any) -> None:
        if tuple(tensor.shape) != self.shape:
            raise LayoutError(f"shape {tuple(tensor.shape)} does not match declared {self.shape}")
        if tensor.dtype != self.dtype or tensor.device != self.device:
            raise LayoutError("tensor dtype/device does not match declared layout")
        if bool(tensor.requires_grad) != self.requires_grad:
            raise LayoutError("tensor requires_grad does not match declared layout")

    def with_shape(self, shape: tuple[int, ...], *, kind: LayoutKind | None = None,
                   shard_dim: int | None = None) -> "TensorLayout":
        return replace(self, shape=shape, kind=kind or self.kind,
                       shard_dim=self.shard_dim if shard_dim is None else shard_dim)


def check_transition(source: TensorLayout, target: TensorLayout) -> None:
    if source.dtype != target.dtype or source.device != target.device:
        raise LayoutError("layout transition cannot change dtype or device")
    if len(source.shape) != len(target.shape):
        raise LayoutError("layout transition cannot change tensor rank")
