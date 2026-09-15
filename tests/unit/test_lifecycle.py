"""ExecutionScheduler contract: drain runs registered work and surfaces failures.

Moved here from ``test_layout_collectives.py`` when ``aitrainer.layout`` was
deleted.  These two tests pin real behaviour -- that ``drain()`` actually invokes
every registered operation, and that a failing wait propagates as
``LifecycleError`` instead of being swallowed -- so they had to move rather than
disappear with the file that happened to host them.
"""

import pytest

from aitrainer.lifecycle import AsyncOp, ExecutionScheduler, LifecycleError


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
