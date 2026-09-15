from aitrainer.overlap import ParameterPrefetchCoordinator


def test_trace_invalidation_falls_back_to_fetch():
    calls = []
    coordinator = ParameterPrefetchCoordinator(fetch_fn=lambda name: calls.append(name) or name)
    coordinator.record("m", ("p",)); assert coordinator.finalize()
    op = coordinator.fetch("p"); assert op.wait() == "p"
    coordinator.invalidate(); assert coordinator.fetch("q") is None
    assert coordinator.wait("q") == "q"
