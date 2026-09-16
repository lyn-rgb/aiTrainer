"""Low-overhead timing, memory sampling, and benchmark artifact helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping

from .metrics import BenchmarkArtifact

# Which profiler events ARE the transfer, and which only sit above it.
#
# `torch.profiler` reports one collective as a nest, measured on two Gloo ranks:
#
#     FSDP::all_gather            user_annotation  dur=217   <- wrapper
#       fsdp::all_gather_copy_in  cpu_op           dur=26    <- local copy, not comms
#       c10d::_allgather_base_    cpu_op           dur=12    <- the issuing call
#       gloo:all_gather           user_annotation  dur=118   <- THE TRANSFER
#       FSDP::all_gather_copy_out user_annotation  dur=42    <- local copy
#
# Matching on the operation name alone counts that once as 217 and again as 118,
# and counts the two copies as communication as well.  The backend prefix is the
# one part of the name that identifies the transfer itself, so that is what this
# matches: everything above it is wrapper, and `*_copy_in`/`*_copy_out` are local
# memory traffic on the critical path.
_COLLECTIVE_BACKENDS = ("gloo:", "nccl:", "mpi:", "xccl:", "ucc:")
_COLLECTIVE_OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all",
                   "alltoall", "broadcast", "send", "recv")
# `c10d::` is deliberately NOT a backend prefix: `c10d::_allgather_base_` is the
# call that ISSUES the transfer, and it is a sibling of the transfer rather than
# its parent, so keeping it would count every collective twice.
#
# Categories whose events represent work the communication could have been
# hidden behind: aten ops on CPU, kernels on CUDA.
_COMPUTE_CATEGORIES = ("cpu_op", "kernel", "Kernel")


def _is_collective(name: str) -> bool:
    return (name.startswith(_COLLECTIVE_BACKENDS)
            and any(op in name for op in _COLLECTIVE_OPS))


@dataclass(frozen=True)
class _Interval:
    """A half-open span on the profiler's microsecond timebase."""

    start: float
    end: float

    def overlap_with(self, other: _Interval) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


def _covered_seconds(target: _Interval, intervals: list[_Interval]) -> float:
    """How much of ``target`` is covered by the union of ``intervals``.

    A union, not a sum: two kernels running concurrently under one collective
    cover the same wall-clock second once.  Summing their durations reported
    overlap above the communication's own duration on the first attempt.
    """
    spans = sorted((max(target.start, i.start), min(target.end, i.end))
                   for i in intervals
                   if i.overlap_with(target) > 0.0)
    covered = 0.0
    current_end = target.start
    for start, end in spans:
        if end <= current_end:
            continue
        covered += end - max(start, current_end)
        current_end = end
    return covered


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    elapsed_seconds: float
    metadata: dict[str, Any]


class Profiler:
    def __init__(self) -> None:
        self.events: list[ProfileEvent] = []
        self._wait_seconds = 0.0
        self._overlapped_seconds = 0.0
        self._pipeline_bubble_seconds = 0.0
        self._gpu_idle_seconds = 0.0
        self.timeline: list[dict[str, Any]] = []
        self._allocation_count = 0
        self._collective_count = 0
        self._buffer_pool_hits = 0
        self._buffer_pool_misses = 0

    def record_async(self, name: str, *, submit_ts: float, producer_ready_ts: float | None = None,
                     wait_start_ts: float | None = None, wait_end_ts: float | None = None,
                     complete_ts: float | None = None, bytes_: int = 0, stream: Any = None,
                     group: Any = None, bucket: str | None = None, module: str | None = None,
                     micro_batch: int | None = None, kind: str = "communication") -> None:
        """Record one operation's complete dependency timeline."""
        complete = complete_ts if complete_ts is not None else time.perf_counter()
        wait_start = wait_start_ts if wait_start_ts is not None else complete
        wait_end = wait_end_ts if wait_end_ts is not None else complete
        ready = producer_ready_ts if producer_ready_ts is not None else submit_ts
        exposed = max(0.0, wait_end - wait_start)
        total = max(0.0, complete - submit_ts)
        hidden = max(0.0, total - exposed)
        self.timeline.append({"name": name, "kind": kind, "submit_ts": submit_ts,
                              "producer_ready_ts": ready, "consumer_wait_start_ts": wait_start,
                              "consumer_wait_end_ts": wait_end, "complete_ts": complete,
                              "bytes": int(bytes_), "stream": str(stream), "group": str(group),
                              "bucket": bucket, "module": module, "micro_batch": micro_batch,
                              "exposed_communication_time": exposed,
                              "hidden_communication_time": hidden,
                              "overlap_ratio": hidden / total if total else 0.0})

    def record_allocation(self, *, pool_hit: bool = False) -> None:
        self._allocation_count += 1
        if pool_hit: self._buffer_pool_hits += 1
        else: self._buffer_pool_misses += 1

    def record_wait(self, seconds: float, *, overlapped_seconds: float = 0.0) -> None:
        """``seconds`` exposed, ``overlapped_seconds`` hidden behind compute.

        The two are independent: ``summary`` reports ``hidden / (exposed + hidden)``,
        so hidden exceeding exposed is not a mistake but the goal -- it is
        communication almost entirely covered by work.

        This used to accumulate ``min(seconds, overlapped_seconds)``, which caps
        hidden at exposed and therefore caps ``overlap_ratio`` at 0.5.  Nothing
        caught it because ``record_wait`` had no caller feeding it real numbers:
        both existing cases passed hidden <= exposed by construction.
        """
        if seconds < 0 or overlapped_seconds < 0:
            raise ValueError("wait and overlap durations must be non-negative")
        self._wait_seconds += seconds
        self._overlapped_seconds += overlapped_seconds

    def record_pipeline_bubble(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("pipeline bubble must be non-negative")
        self._pipeline_bubble_seconds += seconds

    def record_gpu_idle(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("GPU idle duration must be non-negative")
        self._gpu_idle_seconds += seconds

    @contextmanager
    def capture(self, *, trace_path: str | os.PathLike[str] | None = None,
                activities: Any = None) -> Iterator[Any]:
        """Measure the block's communication against the compute it could hide behind.

        ``record_async`` and ``record_wait`` need callers at every collective, and
        there is no such place for most of them: the all-gathers and
        reduce-scatters inside DTensor and FSDP2 are autograd nodes in torch, not
        calls this framework makes.  ``torch.profiler`` sees them all, so this
        wraps a training loop and reads the trace back::

            with trainer.profiler.capture():
                trainer.fit(batches, epochs=1)
            print(trainer.profiler.summary()["overlap_ratio"])

        Opt-in by construction -- profiling costs real time, and a metric that
        silently taxes every run is worse than no metric.

        What it can and cannot say, measured rather than assumed: on CPU/Gloo the
        backend runs its transfer on a worker thread, so the intervals really do
        interleave and ``hidden`` is meaningful there.  The first CPU/Gloo trace
        of an FSDP step showed ``gloo:all_gather`` running on thread 199 while the
        main thread waited, and covered by nothing -- ``exposed`` equalled its
        duration, which is the truth for a caller that waits.  On CUDA the
        transfers and kernels overlap properly and the same arithmetic gives the
        number the overlap work is trying to move.  This host has no CUDA, so that
        half is unverified here.
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError("capture requires PyTorch") from exc
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as profiled:
            yield profiled
        path = Path(trace_path) if trace_path is not None else Path(
            tempfile.mkdtemp(prefix="aitrainer-trace-")) / "trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        profiled.export_chrome_trace(str(path))
        self.record_trace(path)

    def record_trace(self, trace_path: str | os.PathLike[str]) -> dict[str, float]:
        """Fold one chrome trace's collective/compute overlap into this profiler.

        ``exposed`` is the part of each transfer's span that no compute covers --
        the caller is waiting through it.  Everything else is hidden.  Both are
        accumulated through :meth:`record_wait`, so ``summary`` reports them the
        same way whether they came from here or from a hand-placed call.

        Refuses a trace with no collectives: a framework measurably doing
        distributed work that a profiler reports none of means the *naming* was
        wrong, and returning a reassuring ``overlap_ratio`` of 0.0 would hide
        exactly the thing this exists to reveal.
        """
        events = json.loads(Path(trace_path).read_text(encoding="utf-8"))["traceEvents"]
        spans = [event for event in events if event.get("ph") == "X" and event.get("dur")]
        compute = [_Interval(e["ts"], e["ts"] + e["dur"]) for e in spans
                   if e.get("cat") in _COMPUTE_CATEGORIES and not _is_collective(e.get("name", ""))]
        collectives = [event for event in spans if _is_collective(event.get("name", ""))]
        if not collectives:
            raise ValueError(
                f"no collective events in {trace_path}; the backend naming this "
                "matches on "
                f"({', '.join(_COLLECTIVE_BACKENDS)}) may not be what produced this "
                "trace, which means an overlap of 0.0 here would be an artefact")
        exposed_total = hidden_total = 0.0
        for event in collectives:
            span = _Interval(event["ts"], event["ts"] + event["dur"])
            hidden = min(span.end - span.start, _covered_seconds(span, compute))
            hidden_total += hidden
            exposed_total += (span.end - span.start) - hidden
            self.timeline.append({
                "name": event["name"], "kind": "communication",
                "submit_ts": span.start, "complete_ts": span.end,
                "exposed_communication_time": (span.end - span.start) - hidden,
                "hidden_communication_time": hidden,
                "overlap_ratio": hidden / (span.end - span.start) if span.end > span.start else 0.0})
        self._collective_count += len(collectives)
        self.record_wait(exposed_total, overlapped_seconds=hidden_total)
        return {"collectives": float(len(collectives)),
                "exposed_seconds": exposed_total, "hidden_seconds": hidden_total}

    @contextmanager
    def section(self, name: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(ProfileEvent(name, time.perf_counter() - started, metadata))

    def memory_snapshot(self) -> dict[str, int]:
        try:
            import torch
            if torch.cuda.is_available():
                return {"allocated_bytes": int(torch.cuda.memory_allocated()),
                        "reserved_bytes": int(torch.cuda.memory_reserved()),
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        except ImportError:
            return {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0}
        return {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0}

    def summary(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for event in self.events:
            grouped.setdefault(event.name, []).append(event.elapsed_seconds)
        result = {name: sum(values) / len(values) for name, values in grouped.items() if values}
        total = sum(event.elapsed_seconds for event in self.events)
        # The numerator is communication time, so the denominator must be too.
        # Dividing it by section durations produced ratios around 3.5e6, and
        # nested sections were double-counted in that denominator.
        comm_seconds = self._wait_seconds + self._overlapped_seconds
        ratio = (self._overlapped_seconds / comm_seconds) if comm_seconds else 0.0
        result.update({"wait_seconds": self._wait_seconds,
                       "overlapped_seconds": self._overlapped_seconds,
                       "section_seconds": total,
                       "overlap_ratio": min(1.0, max(0.0, ratio)),
                       "pipeline_bubble_seconds": self._pipeline_bubble_seconds,
                       "gpu_idle_seconds": self._gpu_idle_seconds})
        result.update({"timeline_events": float(len(self.timeline)),
                       "collectives": float(self._collective_count),
                       "allocation_count": float(self._allocation_count),
                       "buffer_pool_hit_rate": (self._buffer_pool_hits / (self._buffer_pool_hits + self._buffer_pool_misses)
                                                 if self._buffer_pool_hits + self._buffer_pool_misses else 0.0)})
        return result

    def artifact(self, name: str, *, config: Mapping[str, Any], hardware: Mapping[str, Any],
                 metrics: Mapping[str, float] | None = None, notes: tuple[str, ...] = ()) -> BenchmarkArtifact:
        values = dict(metrics or self.summary())
        values.update({f"memory.{key}": float(value) for key, value in self.memory_snapshot().items()})
        return BenchmarkArtifact.create(name, config=config, hardware=hardware, metrics=values, notes=notes)

    def timeline_artifact(self) -> list[dict[str, Any]]:
        return list(self.timeline)
