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
