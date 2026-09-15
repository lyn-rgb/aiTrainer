"""Kernel capability probing and explicit eager fallback selection.

This module does not import optional CUDA libraries at import time.  A backend
must describe its device/dtype/layout constraints; selection returns a
structured record so callers can persist which implementation actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class KernelStatus(str, Enum):
    AVAILABLE = "available"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"


class KernelSelectionError(RuntimeError):
    """Raised when an explicitly requested kernel cannot be selected."""


@dataclass(frozen=True)
class KernelCapability:
    name: str
    backend: str
    status: KernelStatus
    supported_dtypes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ("cpu", "cuda")
    layouts: tuple[str, ...] = ("contiguous",)
    max_rank: int | None = None
    error_budget: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class KernelSelection:
    name: str
    backend: str
    status: KernelStatus
    callable: Callable[..., Any]
    capability: KernelCapability
    used_fallback: bool = False


class KernelBackend:
    """Registry for one kernel operation and its synchronous eager fallback."""

    def __init__(self, name: str, *, eager: Callable[..., Any]) -> None:
        if not name or not callable(eager):
            raise ValueError("kernel name and eager callable are required")
        self.name = name
        self._eager = eager
        self._implementations: dict[str, tuple[Callable[..., Any], KernelCapability, Callable[..., bool]]] = {}

    def register(self, backend: str, implementation: Callable[..., Any], capability: KernelCapability,
                 probe: Callable[..., bool] | None = None) -> None:
        if backend == "eager":
            raise ValueError(
                "'eager' is the mandatory built-in fallback and cannot be registered; "
                "pass it as KernelBackend(name, eager=...) -- KernelBackend.select() "
                "returns it automatically when no registered backend matches")
        if capability.name != self.name or capability.backend != backend:
            raise ValueError("capability does not match kernel registration")
        self._implementations[backend] = (implementation, capability,
                                          probe or (lambda **_: True))

    @staticmethod
    def _name(value: Any) -> str:
        return str(value).replace("torch.", "").lower()

    def select(self, *, device: Any = "cpu", dtype: Any = "float32", layout: str = "contiguous",
               shape: tuple[int, ...] | None = None, backend: str | None = None,
               allow_fallback: bool = True) -> KernelSelection:
        device_name = str(getattr(device, "type", device)).split(":", 1)[0]
        dtype_name = self._name(dtype)
        for candidate, (implementation, capability, probe) in self._implementations.items():
            if backend is not None and candidate != backend:
                continue
            if device_name not in capability.devices or layout not in capability.layouts:
                continue
            if capability.supported_dtypes and dtype_name not in capability.supported_dtypes:
                continue
            if capability.max_rank is not None and shape is not None and len(shape) > capability.max_rank:
                continue
            if probe(device=device, dtype=dtype, layout=layout, shape=shape):
                return KernelSelection(self.name, candidate, KernelStatus.AVAILABLE, implementation, capability)
        if backend is not None and backend != "eager" and not allow_fallback:
            raise KernelSelectionError(f"kernel {self.name!r} backend {backend!r} is unavailable")
        fallback_capability = KernelCapability(
            self.name, "eager", KernelStatus.FALLBACK,
            supported_dtypes=("float16", "bfloat16", "float32", "float64"),
            devices=("cpu", "cuda"), layouts=("contiguous", "strided"),
            # 0.0 was unattainable: the fallback rounds once in the INPUT dtype, so
            # against a float32 reference its deviation is bounded by that dtype's
            # own precision -- bf16's 2**-8, which matches measurement (~3.9e-3).
            error_budget=2 ** -8,
            reason="mandatory PyTorch eager implementation; exact in the input dtype, "
                   "so deviation from a float32 reference is bounded by the input dtype")
        return KernelSelection(self.name, "eager", KernelStatus.FALLBACK, self._eager,
                               fallback_capability, used_fallback=True)

    def execute(self, *args: Any, backend: str | None = None, allow_fallback: bool = True,
                **kwargs: Any) -> Any:
        value = args[0] if args else kwargs.get("input")
        selection = self.select(device=getattr(value, "device", "cpu"),
                                dtype=getattr(value, "dtype", "float32"),
                                shape=tuple(getattr(value, "shape", ())), backend=backend,
                                allow_fallback=allow_fallback)
        return selection.callable(*args, **kwargs)

    def capabilities(self) -> tuple[KernelCapability, ...]:
        return tuple(capability for _, capability, _ in self._implementations.values())
