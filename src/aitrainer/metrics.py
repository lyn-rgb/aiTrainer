"""Structured training and benchmark metrics with atomic artifact output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping


@dataclass
class MetricStore:
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))

    def latest(self, name: str) -> float | None:
        values = self.values.get(name, [])
        return values[-1] if values else None

    def summary(self) -> dict[str, float]:
        return {name: sum(values) / len(values) for name, values in self.values.items() if values}


@dataclass(frozen=True)
class BenchmarkArtifact:
    name: str
    created_at: float
    config: dict[str, Any]
    hardware: dict[str, Any]
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()

    @classmethod
    def create(cls, name: str, *, config: Mapping[str, Any], hardware: Mapping[str, Any],
               metrics: Mapping[str, float], notes: tuple[str, ...] = ()) -> "BenchmarkArtifact":
        return cls(name, time.time(), dict(config), dict(hardware), dict(metrics), notes)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        import os
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(target)
            return target
        finally:
            if temporary.exists():
                temporary.unlink()
