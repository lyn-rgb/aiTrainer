from aitrainer.diagnostics import diagnose_exception, inspect_checkpoint
from aitrainer.profiler import Profiler


def test_diagnostics_include_rank_context(tmp_path):
    report = diagnose_exception(RuntimeError("NCCL timeout"), rank=2, stage=1, microbatch=3)
    assert report["kind"] == "communication" and report["rank"] == 2
    assert inspect_checkpoint(tmp_path / "missing")["status"] == "missing"


def test_profiler_creates_artifact(tmp_path):
    profiler = Profiler()
    with profiler.section("forward"):
        _ = 1 + 1
    artifact = profiler.artifact("unit", config={}, hardware={})
    path = artifact.write(tmp_path / "artifact.json")
    assert path.is_file() and "forward" in artifact.metrics


def _trace(tmp_path, events):
    """A chrome trace holding exactly ``events``: (name, category, ts, dur)."""
    import json
    payload = {"traceEvents": [
        {"ph": "X", "name": name, "cat": category, "ts": ts, "dur": duration, "tid": 1, "pid": 1}
        for name, category, ts, duration in events]}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_overlap_is_the_part_of_a_transfer_no_compute_covers(tmp_path):
    """The three regimes, on a synthetic trace where the answer is arithmetic.

    A transfer with nothing running under it is fully exposed (that is a caller
    waiting); one fully covered is fully hidden; a partial overlap splits.
    """
    profiler = Profiler()
    profiler.record_trace(_trace(tmp_path, [
        # fully exposed: no compute between 0 and 100
        ("gloo:all_reduce", "user_annotation", 0, 100),
        # fully hidden: compute spans the whole transfer
        ("gloo:all_gather", "user_annotation", 1000, 100),
        ("aten::mm", "cpu_op", 950, 200),
        # half: compute covers the second half only
        ("gloo:reduce_scatter", "user_annotation", 2000, 100),
        ("aten::add", "cpu_op", 2050, 100),
    ]))
    summary = profiler.summary()
    assert summary["overlapped_seconds"] == 150.0        # 0 + 100 + 50
    assert summary["wait_seconds"] == 150.0              # 100 + 0 + 50
    assert summary["overlap_ratio"] == 0.5


def test_concurrent_compute_is_counted_once_not_summed(tmp_path):
    """Two kernels under one transfer cover the same wall-clock second ONCE.

    Summing their durations instead of merging the intervals is the natural way
    to write this and it reported more hidden time than the transfer lasted --
    which then clamps to a ratio of 1.0 and looks like a perfect overlap.
    """
    profiler = Profiler()
    profiler.record_trace(_trace(tmp_path, [
        ("gloo:all_reduce", "user_annotation", 0, 100),
        ("aten::mm", "cpu_op", 0, 60),          # two kernels running side by side,
        ("aten::mm", "cpu_op", 0, 60),          # together covering only 0..60
    ]))
    result = profiler.summary()
    assert result["overlapped_seconds"] == 60.0, "concurrent spans were summed, not merged"
    assert result["wait_seconds"] == 40.0
    assert result["overlap_ratio"] == 0.6


def test_the_wrapper_around_a_collective_is_not_a_second_collective(tmp_path):
    """One transfer is reported as a nest; only the transfer itself is counted.

    Measured on two Gloo ranks, an FSDP all-gather appears as

        FSDP::all_gather            dur=217   <- wrapper
          fsdp::all_gather_copy_in  dur=26    <- local copy, not communication
          c10d::_allgather_base_    dur=12    <- the issuing call, a SIBLING
          gloo:all_gather           dur=118   <- the transfer
          FSDP::all_gather_copy_out dur=42

    Matching on the operation name counts that as 217 + 26 + 12 + 118 + 42.
    """
    profiler = Profiler()
    profiler.record_trace(_trace(tmp_path, [
        ("FSDP::all_gather", "user_annotation", 0, 217),
        ("fsdp::all_gather_copy_in", "cpu_op", 5, 26),
        ("c10d::_allgather_base_", "cpu_op", 40, 12),
        ("gloo:all_gather", "user_annotation", 70, 118),
        ("FSDP::all_gather_copy_out", "user_annotation", 200, 42),
    ]))
    assert profiler.summary()["wait_seconds"] == 118.0, (
        "the wrapper or the copies were counted as communication too")


def test_a_trace_without_collectives_is_refused(tmp_path):
    """A zero must mean "measured zero", never "found nothing to measure".

    If the backend naming changes, the match finds nothing and an
    overlap_ratio of 0.0 would read as "communication is never hidden" -- the
    opposite of the truth, and unfalsifiable from the output.
    """
    import pytest

    profiler = Profiler()
    with pytest.raises(ValueError) as caught:
        profiler.record_trace(_trace(tmp_path, [("aten::mm", "cpu_op", 0, 10)]))
    assert "collective" in str(caught.value)


def test_hidden_time_may_exceed_exposed_time():
    """``overlap_ratio`` has to be able to approach 1.0, or it measures nothing.

    ``record_wait`` accumulated ``min(exposed, hidden)``, which caps the ratio at
    0.5 -- while a ratio near 1.0 is exactly what hiding communication is for.
    No caller ever fed it hidden > exposed, so both existing cases sat under the
    cap and the cap never showed.
    """
    profiler = Profiler()
    profiler.record_wait(1.0, overlapped_seconds=9.0)
    summary = profiler.summary()
    assert summary["overlapped_seconds"] == 9.0, "hidden time was clamped to the exposed time"
    assert summary["overlap_ratio"] == 0.9
