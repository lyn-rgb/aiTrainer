"""Dtype naming and sizing, shared by every layer that needs them.

``str(dtype).replace("torch.", "")`` appeared at twelve sites with two different
policies: three of them also lowercased, the rest did not.  The difference is
invisible for the values that actually reach them -- config strings are validated
against lowercase sets and ``str(torch.float16)`` is always ``"torch.float16"`` --
but "invisible today" is exactly how two conventions become two behaviours.  This
module lowercases, uniformly, which is the more forgiving of the two.
"""

from __future__ import annotations

from typing import Any

# Bytes per element for dtypes that can be named without importing torch.
_DTYPE_BYTES: dict[str, int] = {
    "bool": 1, "uint8": 1, "int8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int16": 2, "float16": 2, "bfloat16": 2,
    "int32": 4, "float32": 4, "int64": 8, "float64": 8,
}

# Longest tokens first so "float64" is not matched by "float16"-style prefixes.
_SIZE_TOKENS: tuple[tuple[str, int], ...] = (
    ("float64", 8), ("int64", 8), ("float32", 4), ("int32", 4),
    ("float16", 2), ("bfloat16", 2), ("int16", 2),
    ("bool", 1), ("uint8", 1), ("int8", 1),
)


def dtype_name(dtype: Any) -> str:
    """Canonical short name: ``"torch.float16"`` and ``"Float16"`` both give ``"float16"``."""
    return str(dtype).replace("torch.", "").lower()


def resolve_dtype_name(dtype: Any, torch_module: Any = None) -> Any:
    """Turn a dtype name into a torch dtype, or return ``dtype`` unchanged.

    Without a torch module only the three names the framework actually supports
    are accepted, as strings -- the behaviour ``precision.resolve_dtype`` already
    had for the torch-absent case.
    """
    if not isinstance(dtype, str):
        return dtype
    name = dtype_name(dtype)
    if torch_module is None:
        from .torch import require_torch
        torch_module = require_torch("PyTorch is required to resolve dtype objects")
    if not hasattr(torch_module, name):
        raise ValueError(f"unsupported dtype {dtype!r}")
    return getattr(torch_module, name)


def dtype_size(dtype: Any) -> int:
    """Best-effort bytes per element, without importing torch.

    Consolidates ``memory.buffers._dtype_size`` (which had the string table and
    the token scan) with the tensor-side ``offload.parameter._itemsize``.
    """
    if isinstance(dtype, str):
        return _DTYPE_BYTES.get(dtype_name(dtype), 4)
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    name = str(dtype).lower()
    for token, size in _SIZE_TOKENS:
        if token in name:
            return size
    return 4
