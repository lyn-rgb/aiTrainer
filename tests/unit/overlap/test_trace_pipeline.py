from aitrainer.overlap import ParameterPrefetchCoordinator, PipelineHandleQueue, PipelineKey
from aitrainer.lifecycle import AsyncOp, LifecycleError


def test_trace_invalidation_falls_back_to_fetch():
    calls = []
    coordinator = ParameterPrefetchCoordinator(fetch_fn=lambda name: calls.append(name) or name)
    coordinator.record("m", ("p",)); assert coordinator.finalize()
    op = coordinator.fetch("p"); assert op.wait() == "p"
    coordinator.invalidate(); assert coordinator.fetch("q") is None
    assert coordinator.wait("q") == "q"


def test_pipeline_queue_detects_duplicate_and_missing_handles():
    queue = PipelineHandleQueue()
    key = PipelineKey(0, 0, 0, "forward")
    queue.post(key, None)
    duplicate = False
    try:
        queue.post(key, None)
        assert False
    except LifecycleError:
        duplicate = True
    assert duplicate
    queue.wait(key)
    missing = False
    try:
        queue.wait(key)
        assert False
    except LifecycleError:
        missing = True
    assert missing
