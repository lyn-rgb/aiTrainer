import sys

import pytest

from aitrainer.config import PrecisionConfig
from aitrainer.precision import PrecisionError, resolve_dtype, validate_precision


def test_precision_dtype_strings_resolve_without_torch(monkeypatch):
    """The torch-free fallback returns the dtype name unchanged.

    Setting ``sys.modules['torch'] = None`` makes ``import torch`` raise
    ImportError, so this exercises the documented fallback in every environment
    instead of only passing on machines that happen to lack torch.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_dtype("float32") == "float32"


def test_precision_dtype_strings_resolve_with_torch():
    torch = pytest.importorskip("torch")
    assert resolve_dtype("float32") is torch.float32


def test_precision_device_is_validated():
    validate_precision(PrecisionConfig(), device="cpu")
    with pytest.raises(PrecisionError):
        validate_precision(PrecisionConfig(), device="quantum0")
