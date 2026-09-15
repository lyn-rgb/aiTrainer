"""Megatron-style sequence-parallel helpers built on the TP group."""

from __future__ import annotations

from typing import Any

from ..core.torch import module_base
from .collectives import all_reduce_gradient, gather_from_sequence, scatter_to_sequence

# Must be an expression at import time: the classes below inherit from it.
ModuleBase, nn = module_base()


def scatter_sequence(value: Any, group: Any = None, *, dim: int = 1) -> Any:
    return scatter_to_sequence(value, group, dim=dim)


def gather_sequence(value: Any, group: Any = None, *, dim: int = 1) -> Any:
    return gather_from_sequence(value, group, dim=dim)


class SequenceParallelLayerNorm(ModuleBase):
    """LayerNorm over hidden features while sequence is sharded across TP ranks."""

    def __init__(self, normalized_shape: Any, *, eps: float = 1e-5,
                 elementwise_affine: bool = True, process_group: Any = None,
                 input_is_parallel: bool = False, gather_output: bool = False) -> None:
        if nn is None:  # type: ignore[truthy-function]
            raise ImportError("SequenceParallelLayerNorm requires PyTorch")
        super().__init__()
        self.process_group = process_group
        self.input_is_parallel = input_is_parallel
        self.gather_output = gather_output
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        # weight/bias are REPLICATED across the TP group, but each rank's
        # gradient is computed from only its own token shard.  Without this
        # reduction the copies diverge after the first optimizer step (the
        # forward output and the INPUT gradient were already correct).
        if elementwise_affine and process_group is not None:
            self.norm.weight.register_hook(
                lambda grad: all_reduce_gradient(grad, self.process_group))
            if self.norm.bias is not None:
                self.norm.bias.register_hook(
                    lambda grad: all_reduce_gradient(grad, self.process_group))

    def forward(self, value: Any) -> Any:
        if not self.input_is_parallel:
            value = scatter_sequence(value, self.process_group)
        value = self.norm(value)
        return gather_sequence(value, self.process_group) if self.gather_output else value


def sequence_parallel_dropout(value: Any, dropout: Any, *, process_group: Any = None,
                              input_is_parallel: bool = True, gather_output: bool = False) -> Any:
    if not input_is_parallel:
        value = scatter_sequence(value, process_group)
    value = dropout(value)
    return gather_sequence(value, process_group) if gather_output else value
