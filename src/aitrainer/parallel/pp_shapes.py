"""Pipeline tensor metadata, stage planning, and micro-batch utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


class PipelineShapeError(ValueError):
    """Raised when a pipeline tensor or stage plan is inconsistent."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    stage: int
    microbatch: int
    tag: str

    def __post_init__(self) -> None:
        if not self.shape or any(int(size) < 0 for size in self.shape):
            raise PipelineShapeError(f"invalid pipeline tensor shape {self.shape}")
        if self.stage < 0 or self.microbatch < 0 or not self.tag:
            raise PipelineShapeError("stage, microbatch and tag must be explicit")

    @classmethod
    def from_tensor(cls, tensor: Any, *, stage: int, microbatch: int, tag: str) -> "TensorSpec":
        return cls(tuple(tensor.shape), tensor.dtype, tensor.device, stage, microbatch, tag)

    def validate_tensor(self, tensor: Any) -> None:
        if tuple(tensor.shape) != self.shape:
            raise PipelineShapeError(f"{self.tag}: shape {tuple(tensor.shape)} != expected {self.shape}")
        if tensor.dtype != self.dtype:
            raise PipelineShapeError(f"{self.tag}: dtype {tensor.dtype} != expected {self.dtype}")


@dataclass(frozen=True)
class StagePlan:
    stage: int
    start: int
    stop: int
    parameter_count: int
    estimated_flops: float
    activation_elements: int

    @property
    def num_layers(self) -> int:
        return self.stop - self.start


def _layer_cost(layer: Any, sample: Any = None) -> tuple[int, float, int]:
    parameters = sum(int(p.numel()) for p in layer.parameters()) if hasattr(layer, "parameters") else 0
    flops = float(parameters * 2)
    activation = int(sample.numel()) if hasattr(sample, "numel") else 0
    return parameters, flops, activation


def plan_stages(layers: Sequence[Any], pp_size: int, *, policy: str = "uniform_layers",
                sample: Any = None, cost_fn: Callable[[Any], tuple[int, float, int]] | None = None
                ) -> tuple[StagePlan, ...]:
    """Create a deterministic non-empty stage plan for a sequential layer list."""
    if pp_size < 1 or pp_size > len(layers):
        raise PipelineShapeError(f"pp_size={pp_size} requires 1 <= pp_size <= num_layers={len(layers)}")
    if policy not in {"uniform_layers", "parameter_count", "flops", "memory_balanced"}:
        raise PipelineShapeError(f"unknown stage split policy {policy!r}")
    costs = [cost_fn(layer) if cost_fn else _layer_cost(layer, sample) for layer in layers]
    boundaries: list[int] = [0]
    if policy == "uniform_layers":
        for stage in range(1, pp_size):
            boundaries.append(round(stage * len(layers) / pp_size))
    else:
        values = [item[0] if policy == "parameter_count" else item[1] if policy == "flops"
                  else item[0] + item[2] for item in costs]
        if sum(values) == 0:
            for stage in range(1, pp_size):
                boundaries.append(round(stage * len(layers) / pp_size))
        else:
            target = sum(values) / pp_size
            for stage in range(1, pp_size):
                goal = target * stage
                boundary = max(boundaries[-1] + 1, min(len(layers) - (pp_size - stage),
                                                       next((i for i, _ in enumerate(values, 1)
                                                             if sum(values[:i]) >= goal), len(layers) - 1)))
                boundaries.append(boundary)
    boundaries.append(len(layers))
    plans = []
    for stage, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if stop <= start:
            raise PipelineShapeError(f"stage {stage} is empty: [{start}, {stop})")
        selected = costs[start:stop]
        plans.append(StagePlan(stage, start, stop, sum(item[0] for item in selected),
                               sum(item[1] for item in selected), sum(item[2] for item in selected)))
    return tuple(plans)


def split_sequential(module: Any, pp_size: int, *, policy: str = "uniform_layers",
                     sample: Any = None) -> tuple[Any, tuple[StagePlan, ...]]:
    """Split a module with an ordered ``children()`` contract into stages."""
    import torch.nn as nn
    layers = list(module.children())
    plans = plan_stages(layers, pp_size, policy=policy, sample=sample)
    stages = tuple(nn.Sequential(*layers[p.start:p.stop]) for p in plans)
    return stages, plans


def split_microbatches(batch: Any, count: int, *, batch_dim: int = 0) -> list[Any]:
    """Split tensor/mapping/tuple batches evenly without changing their contract."""
    if count < 1:
        raise PipelineShapeError("microbatch count must be positive")
    if hasattr(batch, "shape") and hasattr(batch, "split"):
        size = batch.shape[batch_dim]
        if size % count:
            raise PipelineShapeError(f"batch dimension {size} is not divisible by microbatches={count}")
        return list(batch.split(size // count, dim=batch_dim))
    if isinstance(batch, Mapping):
        parts = {key: split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for key, value in batch.items()}
        return [{key: values[index] for key, values in parts.items()} for index in range(count)]
    if isinstance(batch, tuple):
        parts = [split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for value in batch]
        return [tuple(values[index] for values in parts) for index in range(count)]
    if isinstance(batch, list):
        parts = [split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for value in batch]
        return [[values[index] for values in parts] for index in range(count)]
    raise PipelineShapeError(f"cannot split batch type {type(batch).__name__}")


def normalize_loss(loss: Any, *, valid_tokens: int | None = None) -> Any:
    """Normalize a scalar or token-summed loss exactly once."""
    if not hasattr(loss, "ndim") or loss.ndim != 0:
        raise PipelineShapeError("pipeline loss must be scalar")
    if valid_tokens is not None:
        if valid_tokens < 1:
            raise PipelineShapeError("valid_tokens must be positive")
        return loss / valid_tokens
    return loss
