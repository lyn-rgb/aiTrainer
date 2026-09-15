"""Megatron-style synchronous Column and Row Parallel Linear layers."""

from __future__ import annotations

from typing import Any

from ..core.torch import module_base
from ..core.torch import rank as _core_rank
from ..core.torch import world_size as _core_world_size

ModuleBase, nn = module_base()


class TPConfigurationError(ValueError):
    """Raised when a TP dimension or layout contract is invalid."""


def _require_divisible(value: int, world: int, name: str) -> int:
    if value % world:
        raise TPConfigurationError(f"{name}={value} must be divisible by TP world size {world}")
    return value // world


class ColumnParallelLinear(ModuleBase):
    """Linear layer with output dimension partitioned across the TP group."""

    def __init__(self, input_size: int, output_size: int, *, bias: bool = True,
                 gather_output: bool = False, skip_bias_add: bool = False,
                 process_group: Any = None, init_method: Any = None,
                 overlap_controller: Any = None) -> None:
        if nn is None:  # type: ignore[truthy-function]
            raise ImportError("ColumnParallelLinear requires PyTorch")
        import torch
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.process_group = process_group
        self.overlap_controller = overlap_controller
        self.tp_size = _core_world_size(process_group)
        self.tp_rank = _core_rank(process_group) if self.tp_size > 1 else 0
        self.output_size_per_partition = _require_divisible(output_size, self.tp_size, "output_size")
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method: Any = None) -> None:
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = self.weight.shape[1] ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: Any, *, input_is_parallel: bool = False) -> Any:
        import torch

        from .collectives import copy_to_tp, gather_from_tp
        if not input_is_parallel:
            if self.overlap_controller is not None:
                value = self.overlap_controller.submit("tp.forward_input_copy", lambda: copy_to_tp(value, self.process_group))
            else:
                value = copy_to_tp(value, self.process_group)
        output = torch.nn.functional.linear(value, self.weight, None)
        # Bias must be added while the output is still SHARDED: self.bias holds
        # output_size_per_partition entries, so adding it after gathering (when the
        # output is output_size wide) mismatched by a factor of tp_size.  Summing
        # each shard's bias before concatenation is equivalent to adding the full
        # bias, and keeps the same numerics.
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias
        if self.gather_output:
            output = gather_from_tp(output, self.process_group, dim=-1)
            if self.skip_bias_add and self.bias is not None:
                # The caller fuses the bias add downstream, so it needs a bias that
                # matches the gathered width, not the shard width.
                return output, gather_from_tp(self.bias, self.process_group, dim=-1)
            return output
        if self.skip_bias_add:
            return output, self.bias
        return output

    def shard_metadata(self) -> dict[str, Any]:
        return {"partition_dim": 0, "global_shape": (self.output_size, self.input_size),
                "local_shape": tuple(self.weight.shape), "tp_rank": self.tp_rank,
                "tp_size": self.tp_size}

    @classmethod
    def from_dense(cls, dense: Any, *, process_group: Any = None, **kwargs: Any) -> ColumnParallelLinear:
        import torch
        layer = cls(dense.in_features, dense.out_features, bias=dense.bias is not None,
                    process_group=process_group, **kwargs)
        rank = _core_rank(process_group) if _core_world_size(process_group) > 1 else 0
        start = rank * layer.output_size_per_partition
        with torch.no_grad():
            layer.weight.copy_(dense.weight[start:start + layer.output_size_per_partition])
            if layer.bias is not None and dense.bias is not None:
                layer.bias.copy_(dense.bias[start:start + layer.output_size_per_partition])
        return layer


class RowParallelLinear(ModuleBase):
    """Linear layer with input dimension partitioned across the TP group."""

    def __init__(self, input_size: int, output_size: int, *, bias: bool = True,
                 input_is_parallel: bool = False, skip_bias_add: bool = False,
                 process_group: Any = None, init_method: Any = None,
                 reduce_dtype: Any | None = None, overlap_controller: Any = None) -> None:
        if nn is None:  # type: ignore[truthy-function]
            raise ImportError("RowParallelLinear requires PyTorch")
        import torch
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.process_group = process_group
        self.overlap_controller = overlap_controller
        self.tp_size = _core_world_size(process_group)
        self.tp_rank = _core_rank(process_group) if self.tp_size > 1 else 0
        self.input_size_per_partition = _require_divisible(input_size, self.tp_size, "input_size")
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.reduce_dtype = reduce_dtype
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method: Any = None) -> None:
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = self.input_size_per_partition ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: Any, *, input_is_parallel: bool | None = None) -> Any:
        import torch

        from .collectives import reduce_from_tp, scatter_to_tp
        is_parallel = self.input_is_parallel if input_is_parallel is None else input_is_parallel
        if not is_parallel:
            value = scatter_to_tp(value, self.process_group, dim=-1)
        output = torch.nn.functional.linear(value, self.weight, None)
        if self.overlap_controller is not None:
            output = self.overlap_controller.submit("tp.backward_dgrad_reduce", lambda: reduce_from_tp(output, self.process_group, reduce_dtype=self.reduce_dtype))
        else:
            output = reduce_from_tp(output, self.process_group, reduce_dtype=self.reduce_dtype)
        if self.skip_bias_add:
            return output, self.bias
        return output if self.bias is None else output + self.bias

    def shard_metadata(self) -> dict[str, Any]:
        return {"partition_dim": 1, "global_shape": (self.output_size, self.input_size),
                "local_shape": tuple(self.weight.shape), "tp_rank": self.tp_rank,
                "tp_size": self.tp_size}

    @classmethod
    def from_dense(cls, dense: Any, *, process_group: Any = None, **kwargs: Any) -> RowParallelLinear:
        import torch
        layer = cls(dense.in_features, dense.out_features, bias=dense.bias is not None,
                    process_group=process_group, **kwargs)
        rank = _core_rank(process_group) if _core_world_size(process_group) > 1 else 0
        start = rank * layer.input_size_per_partition
        with torch.no_grad():
            layer.weight.copy_(dense.weight[:, start:start + layer.input_size_per_partition])
            if layer.bias is not None and dense.bias is not None:
                layer.bias.copy_(dense.bias)
        return layer
