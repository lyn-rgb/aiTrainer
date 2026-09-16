"""Conservative Transformer TP plan based on explicit module names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def styles(self, module: Any) -> dict[str, Any]:
        """``{module fqn: torch TP style}`` for every projection this plan names.

        The projection layouts do not depend on sequence parallelism.  They
        could: torch's Megatron arrangement pairs a column projection emitting
        ``Shard(1)`` with a row projection reading ``Shard(1)``, which keeps the
        activation between them sharded too.  That requires the *whole residual
        stream* to be sequence-sharded -- the model must be written for it, and
        a plain ``nn.Linear`` in the middle cannot even compute, because
        ``F.linear`` flattens every dimension but the last.  This framework
        cannot install that on an arbitrary ``nn.Module``, so it does not
        pretend to: sequence parallelism here shards each norm's own activation
        and gathers it back (see ``parallel.tp._sharded_norm_style``), which is
        what the previous hand-written implementation did too.
        """
        import torch
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        styles: dict[str, Any] = {}
        for name, child in module.named_modules():
            if not name or not isinstance(child, torch.nn.Linear):
                continue
            role = self.role(name)
            if role == "column":
                styles[name] = ColwiseParallel(output_layouts=Shard(-1))
            elif role == "row":
                styles[name] = RowwiseParallel()
        return styles

    def validate_dimensions(self, module: Any, *, tp_size: int) -> None:
        import torch
        for name, child in module.named_modules():
            if not isinstance(child, torch.nn.Linear):
                continue
            role = self.role(name)
            if role == "column" and child.out_features % tp_size:
                raise ValueError(f"{name}.out_features={child.out_features} is not divisible by tp_size={tp_size}")
            if role == "row" and child.in_features % tp_size:
                raise ValueError(f"{name}.in_features={child.in_features} is not divisible by tp_size={tp_size}")

    def partition_dims(self, module: Any) -> dict[str, int]:
        """``{parameter name: dimension}`` for the tensors tensor parallelism splits.

        ``CheckpointConverter.convert`` needs this to write a genuinely sharded
        checkpoint; without it every tensor is treated as replicated, every rank
        reads the whole model, and the load is correct but saves no IO at all --
        measured at tp=2 as ``bytes_read`` equal to the full model on both ranks.

        It mirrors what torch's styles actually do (read from
        ``torch.distributed.tensor.parallel.style``): a column projection shards
        its weight *and* bias on dim 0, a row projection shards its weight on
        dim 1 and **replicates** its bias.  Deriving it from the plan rather than
        asking the caller keeps the checkpoint and the model from disagreeing
        about which axis was split.
        """
        import torch
        dims: dict[str, int] = {}
        for name, child in module.named_modules():
            if not name or not isinstance(child, torch.nn.Linear):
                continue
            role = self.role(name)
            if role == "column":
                dims[f"{name}.weight"] = 0
                if child.bias is not None:
                    dims[f"{name}.bias"] = 0
            elif role == "row":
                dims[f"{name}.weight"] = 1
                # A row projection's bias is Replicate: splitting it would leave
                # each rank adding a fraction of the bias.
        return dims
