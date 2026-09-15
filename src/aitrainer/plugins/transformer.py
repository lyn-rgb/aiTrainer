"""Conservative Transformer TP plan based on explicit module names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..parallel.tp import ColumnParallelLinear, RowParallelLinear


@dataclass(frozen=True)
class TransformerTPPlan:
    """Default projection roles; no string guessing beyond declared suffixes."""
    column_suffixes: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "linear1")
    row_suffixes: tuple[str, ...] = ("o_proj", "out_proj", "down_proj", "linear2")

    def role(self, name: str) -> str | None:
        leaf = name.rsplit(".", 1)[-1]
        if leaf in self.column_suffixes:
            return "column"
        if leaf in self.row_suffixes:
            return "row"
        return None

    def validate_dimensions(self, module: Any, *, tp_size: int) -> None:
        for name, child in module.named_modules():
            if child.__class__.__name__ != "Linear":
                continue
            role = self.role(name)
            if role == "column" and child.out_features % tp_size:
                raise ValueError(f"{name}.out_features={child.out_features} is not divisible by tp_size={tp_size}")
            if role == "row" and child.in_features % tp_size:
                raise ValueError(f"{name}.in_features={child.in_features} is not divisible by tp_size={tp_size}")


def replace_linear_modules(module: Any, *, process_group: Any = None,
                           plan: TransformerTPPlan | None = None,
                           reduce_dtype: Any = None) -> Any:
    """Replace only explicitly named Linear leaves and preserve dense weights.

    ``reduce_dtype`` is threaded into the row projections so the TP all-reduce can
    run in an accumulation dtype; ``PrecisionConfig.reduce_dtype`` had no consumer
    without this and was silently ignored.
    """
    plan = plan or TransformerTPPlan()
    tp_size = 1
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            tp_size = dist.get_world_size(process_group)
    except ImportError:
        tp_size = 1
    plan.validate_dimensions(module, tp_size=tp_size)
    for name, child in list(module.named_children()):
        full_name = name
        role = plan.role(full_name)
        # Resolve the full path for nested modules before applying the role.
        if role is None:
            for parent_name, candidate in module.named_modules():
                if candidate is child:
                    role = plan.role(parent_name)
                    break
        if role == "column":
            replacement = ColumnParallelLinear.from_dense(child, process_group=process_group)
            setattr(module, name, replacement)
        elif role == "row":
            # A declared Transformer row projection consumes the local output
            # shard of the preceding column projection.  Marking it parallel
            # avoids scattering that already-sharded activation a second time.
            replacement = RowParallelLinear.from_dense(child, process_group=process_group,
                                                        input_is_parallel=True,
                                                        reduce_dtype=reduce_dtype)
            setattr(module, name, replacement)
        else:
            replace_linear_modules(child, process_group=process_group, plan=plan,
                                   reduce_dtype=reduce_dtype)
    return module
