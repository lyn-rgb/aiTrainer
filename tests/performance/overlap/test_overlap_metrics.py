from aitrainer.profiler import Profiler


def test_timeline_exposes_hidden_communication_and_resource_metrics():
    profiler = Profiler()
    profiler.record_async("all_gather", submit_ts=1.0, producer_ready_ts=1.1,
                          wait_start_ts=1.2, wait_end_ts=1.3, complete_ts=1.5,
                          bytes_=1024, kind="tp")
    profiler.record_allocation(pool_hit=True)
    summary = profiler.summary()
    assert summary["timeline_events"] == 1.0
    assert summary["buffer_pool_hit_rate"] == 1.0
    assert profiler.timeline_artifact()[0]["hidden_communication_time"] > 0
