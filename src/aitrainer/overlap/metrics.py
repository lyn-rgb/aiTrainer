"""Structured overlap metrics and resource counters."""
from dataclasses import dataclass, field
from typing import Any
@dataclass
class OverlapMetrics:
    events: list[dict[str, Any]] = field(default_factory=list); inflight_peak: int = 0; inflight_bytes_peak: int = 0; prefetch_hit: int = 0; prefetch_miss: int = 0; prefetch_late: int = 0; prefetch_evicted: int = 0
    def record(self, **event: Any) -> None: self.events.append(event); self.inflight_peak=max(self.inflight_peak, int(event.get("inflight_ops", 0))); self.inflight_bytes_peak=max(self.inflight_bytes_peak, int(event.get("inflight_bytes", 0)))
    def record_prefetch(self, *, hits: int = 0, misses: int = 0, late: int = 0, evicted: int = 0) -> None:
        """Adopt cumulative prefetch counters, e.g. from ParameterPrefetchCoordinator.

        Nothing used to write these four fields, so summary() reported structural
        zeros that looked like measurements.
        """
        self.prefetch_hit, self.prefetch_miss = int(hits), int(misses)
        self.prefetch_late, self.prefetch_evicted = int(late), int(evicted)
    def summary(self) -> dict[str, Any]: return {"events": len(self.events), "inflight_ops_peak": self.inflight_peak, "inflight_bytes_peak": self.inflight_bytes_peak, "prefetch_hit": self.prefetch_hit, "prefetch_miss": self.prefetch_miss, "prefetch_late": self.prefetch_late, "prefetch_evicted": self.prefetch_evicted}
