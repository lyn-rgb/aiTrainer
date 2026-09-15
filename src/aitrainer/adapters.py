"""Stable protocols injected by applications."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

try:
    import torch
    from torch import nn
except ImportError:  # import-time support for config/documentation tooling
    torch = None  # type: ignore[assignment]
    nn = Any  # type: ignore[misc,assignment]


@runtime_checkable
class ModelAdapter(Protocol):
    def build(self, *, device: Any = "cpu", dtype: Any | None = None) -> nn.Module: ...
    def layers(self, model: nn.Module) -> Sequence[nn.Module]: ...
    def estimate_layer_cost(self, layer: nn.Module, input_spec: object) -> object: ...
    def split_stage(self, model: nn.Module, *, policy: object) -> nn.Module: ...
    def input_spec(self) -> object: ...
    def output_for_loss(self, output: object, batch: object) -> object: ...


@runtime_checkable
class DataProvider(Protocol):
    def train_dataloader(self, *, dp_group: object = None, seed: int = 42) -> Iterable[object]: ...


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, model_output: object, batch: object) -> Any: ...


@runtime_checkable
class OptimizerFactory(Protocol):
    def build(self, parameters: Iterable[Any]) -> Any: ...
