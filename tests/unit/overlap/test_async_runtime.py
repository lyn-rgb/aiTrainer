from aitrainer.core.lifecycle import AsyncOp, AsyncState, ExecutionScheduler
from aitrainer.overlap import Backpressure, GradientBucket, PingPongBuffer, ResourceBudget


def test_async_state_machine_and_idempotent_release():
    called = []
    op = AsyncOp("x", _wait_fn=lambda _: called.append("wait"))
    assert op.state is AsyncState.CREATED
    op.submit(); op.wait(); op.wait(); op.release(); op.release()
    assert op.state is AsyncState.RELEASED and called == ["wait"]


def test_scheduler_reverse_drain_and_failure_context():
    order = []
    scheduler = ExecutionScheduler()
    scheduler.register(AsyncOp("a", _wait_fn=lambda _: order.append("a")))
    scheduler.register(AsyncOp("b", _wait_fn=lambda _: order.append("b")))
    scheduler.drain()
    assert order == ["b", "a"] and not scheduler.pending


def test_ping_pong_generation_and_budget():
    pool = PingPongBuffer((bytearray(2), bytearray(2)))
    first = pool.acquire("first")
    second = pool.acquire("second")
    pool.release(first); pool.release(second)
    assert pool.acquire("reused").generation == 2
    budget = Backpressure(ResourceBudget(max_inflight_ops=1))
    op = AsyncOp("budget").submit(); budget.admit(op); budget.drain()


def test_bucket_boundary_does_not_drop_tensor():
    bucket = GradientBucket("g", 4)
    assert bucket.add(b"ab")
    assert not bucket.add(b"abcd")
    assert bucket.flush() == (b"ab",)
