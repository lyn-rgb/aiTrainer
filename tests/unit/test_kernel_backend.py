import pytest

from aitrainer.kernels import KernelBackend, KernelCapability, KernelStatus


def test_kernel_backend_has_explicit_eager_fallback():
    backend = KernelBackend("toy", eager=lambda value: value + 1)
    selection = backend.select(device="cpu", dtype="float32")
    assert selection.backend == "eager"
    assert selection.status == KernelStatus.FALLBACK
    assert selection.callable(2) == 3


def test_kernel_backend_rejects_unavailable_explicit_backend():
    backend = KernelBackend("toy", eager=lambda value: value)
    backend.register("cuda", lambda value: value, KernelCapability(
        "toy", "cuda", KernelStatus.AVAILABLE, devices=("cuda",), supported_dtypes=("float32",)),
        probe=lambda **_: False)
    with pytest.raises(RuntimeError):
        backend.select(device="cuda", dtype="float32", backend="cuda", allow_fallback=False)
