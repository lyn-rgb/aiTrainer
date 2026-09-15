import pytest

from aitrainer.layout import LayoutError, TensorLayout
from aitrainer.lifecycle import AsyncOp, ExecutionScheduler, LifecycleError


def test_layout_contract_rejects_invalid_shard_dimension():
    with pytest.raises(LayoutError):
        TensorLayout("hidden", (2, 3), "float32", "cpu")


def test_scheduler_drains_registered_operations():
    scheduler = ExecutionScheduler()
    called = []
    scheduler.register(AsyncOp("test", _wait_fn=lambda _: called.append(True)))
    scheduler.drain()
    assert called == [True] and not scheduler.pending


def test_scheduler_reports_wait_failure():
    scheduler = ExecutionScheduler()
    scheduler.register(AsyncOp("test", _wait_fn=lambda _: (_ for _ in ()).throw(RuntimeError("boom"))))
    with pytest.raises(LifecycleError):
        scheduler.drain()
