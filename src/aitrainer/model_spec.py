"""Minimal sequential model specification for pipeline stage construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .parallel.pp_shapes import StagePlan, split_sequential


@dataclass(frozen=True)
class SequentialModelSpec:
    layers: Sequence[Any]
    input_spec: Any = None
    output_spec: Any = None

    def stage_modules(self, pp_size: int, *, policy: str = "uniform_layers") -> tuple[tuple[Any, ...], tuple[StagePlan, ...]]:
        import torch.nn as nn
        module = nn.Sequential(*self.layers)
        stages, plans = split_sequential(module, pp_size, policy=policy)
        return stages, plans
