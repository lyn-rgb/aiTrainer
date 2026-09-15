"""Low-overhead timing, memory sampling, and benchmark artifact helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Iterator, Mapping

from .metrics import BenchmarkArtifact


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
        if seconds < 0 or overlapped_seconds < 0:
            raise ValueError("wait and overlap durations must be non-negative")
        self._wait_seconds += seconds
        self._overlapped_seconds += min(seconds, overlapped_seconds)

    def record_pipeline_bubble(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("pipeline bubble must be non-negative")
        self._pipeline_bubble_seconds += seconds

    def record_gpu_idle(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("GPU idle duration must be non-negative")
        self._gpu_idle_seconds += seconds

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
