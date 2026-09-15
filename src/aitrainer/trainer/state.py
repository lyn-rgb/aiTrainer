"""The value a single training step reports back."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepOutput:
    loss: float
    step: int
    optimizer_step: bool
    grad_norm: float | None = None
