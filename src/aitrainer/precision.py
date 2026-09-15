"""Runtime helpers for independent parameter/compute/gradient/reduce dtypes."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable

from .config import PrecisionConfig


class PrecisionError(ValueError):
    """Raised when a requested dtype cannot be represented safely."""


def resolve_dtype(dtype: Any, torch_module: Any | None = None) -> Any:
    torch = torch_module
    if torch is None:
        try:
            import torch as torch_import
            torch = torch_import
        except ImportError as exc:
            if isinstance(dtype, str) and dtype in {"float16", "bfloat16", "float32"}:
                return dtype
            raise PrecisionError("PyTorch is required to resolve dtype objects") from exc
    if isinstance(dtype, str):
        name = dtype.replace("torch.", "")
        if not hasattr(torch, name):
            raise PrecisionError(f"unsupported dtype {dtype!r}")
        return getattr(torch, name)
    return dtype


def validate_precision(config: PrecisionConfig, *, device: Any = "cpu", torch_module: Any | None = None) -> None:
    config.validate()
    device_name = str(getattr(device, "type", device)).split(":", 1)[0]
    if device_name not in {"cpu", "cuda", "xpu", "mps"}:
        raise PrecisionError(f"unsupported precision device {device_name!r}")


def autocast_context(config: PrecisionConfig, device: Any, *, enabled: bool = True) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise PrecisionError("PyTorch is required for autocast") from exc
    if not enabled:
        return nullcontext()
    name = str(config.compute_dtype).replace("torch.", "").lower()
    device_name = str(getattr(device, "type", device)).split(":", 1)[0]
    if device_name == "cuda" and name in {"float16", "bfloat16"}:
        dtype = resolve_dtype(name, torch)
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type=device_name, enabled=False)


def cast_gradients(parameters: Iterable[Any], dtype: Any, *, torch_module: Any | None = None) -> int:
    """Cast existing gradients in-place and return the number of elements changed."""
    resolved = resolve_dtype(dtype, torch_module)
    changed = 0
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None or getattr(gradient, "dtype", None) == resolved:
            continue
        parameter_dtype = getattr(parameter, "dtype", None)
        if parameter_dtype is not None and parameter_dtype != resolved:
            raise PrecisionError(
                f"gradient dtype {resolved!r} differs from parameter dtype {parameter_dtype!r}; "
                "an explicit master-parameter optimizer is required"
            )
        parameter.grad = gradient.to(dtype=resolved)
        changed += int(gradient.numel())
    return changed


def cast_optimizer_state(optimizer: Any, dtype: Any, *, torch_module: Any | None = None) -> int:
    """Cast floating optimizer state tensors while preserving integer counters."""
    resolved = resolve_dtype(dtype, torch_module)
    changed = 0
    for state in getattr(optimizer, "state", {}).values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if key == "step":
                # Adam/AdamW hold the bias-correction counter as a FLOAT32 tensor,
                # so the dtype filter below used to cast it -- contradicting this
                # function's own "preserving integer counters" contract, losing
                # integer exactness past 2**8, and breaking fused CUDA Adam kernels
                # that assert on the step tensor's dtype.
                continue
            if hasattr(value, "dtype") and hasattr(value, "to"):
                name = str(value.dtype).replace("torch.", "")
                target_name = str(resolved).replace("torch.", "")
                if name.startswith(("float", "bfloat")) and name != target_name:
                    state[key] = value.to(dtype=resolved)
                    changed += int(value.numel())
    return changed
