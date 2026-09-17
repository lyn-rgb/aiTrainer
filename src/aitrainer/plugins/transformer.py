"""Conservative Transformer TP plan based on explicit module names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformerTPPlan:
    """Default projection roles; no string guessing beyond declared suffixes."""

    column_suffixes: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "linear1")
    row_suffixes: tuple[str, ...] = ("o_proj", "out_proj", "down_proj", "linear2")
    # Whether a COLUMN projection hands the model a ``DTensor`` that still
    # carries its placement, or a plain local tensor.  Row projections always
    # return a plain tensor; see ``styles`` for why the rule is asymmetric and
    # for the measurement that settled it.
    keep_placements: bool = True

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

        ``keep_placements`` decides whether a projection hands the model a
        ``DTensor`` or a plain local tensor, and it is not a performance knob --
        it decides which models can be sharded at all.

        torch defaults ``use_local_output=True``, which redistributes the output
        to the requested placement and then throws the placement away with
        ``to_local()``.  Measured on a Llama-shaped block at ``tp_size=2``: the
        weight came back ``DTensor(32, 32) local=(16, 32) placements=(Shard(0),)``
        -- correctly sharded -- while ``q_proj(hidden)`` came back
        ``Tensor(2, 8, 16)``.  Sixteen features where the model's own weights
        say thirty-two, and nothing in the tensor says why.  The model then
        fails at ``.view(2, 8, 4, 8)``, an error about reshape sizes that names
        neither tensor parallelism nor the plan, and it CANNOT defend itself:
        recovering would mean dividing the head count by a ``tp_size`` it has no
        way to see.

        Why this never showed up before: under the local arrangement the
        activation between a column and a row projection is each rank's slice,
        and ``RowwiseParallel`` annotates a plain tensor as ``Shard(-1)`` while
        ``ColwiseParallel`` annotates one as ``Replicate()``.  Those two agree,
        so a chain of projections and elementwise ops -- which is what every TP
        test in this repository uses -- is exactly right.  It breaks the moment
        the model needs to know the SHAPE of what it is holding, which is any
        attention with a head reshape.

        Keeping the placement makes the model's own arithmetic the thing that
        carries the sharding: ``.view(B, L, heads, head_dim)`` propagates
        ``Shard(-1) -> Shard(2)`` when the head boundary lines up with the shard
        boundary, and DTensor raises where it cannot.  Set this to False to get
        the old local behaviour back, but then every model must be written for
        it.

        **The rule is asymmetric, and the first version of this got it wrong.**
        It kept placements on BOTH projections, which broke the residual stream
        at the first add::

            hidden = hidden + self.self_attn(self.input_layernorm(hidden))
            RuntimeError: aten.add.Tensor: got mixed torch.Tensor and DTensor,
            need to convert all torch.Tensor to DTensor before calling
            distributed operators!

        ``hidden`` is the embedding's output -- no projection touches it, so it
        is a plain tensor -- while the row projection was handing back a
        ``DTensor``.  A residual stream cannot be half-sharded: the embedding
        and the norms are not, so a projection that returns a DTensor forces
        every one of them to become parallelism-aware too.

        So a **column** projection keeps its placement -- its output is what the
        model reshapes, and the model is the only thing that knows the reshape
        is a head split rather than a feature split -- and a **row** projection
        returns a plain tensor, because it is the one that goes back into the
        residual stream and it has already been reduced to ``Replicate`` by
        then, so ``to_local()`` is lossless.

        That is also why no model-side change is needed beyond the ops DTensor
        cannot dispatch: the sharding stays inside the attention/MLP region and
        the residual stream never sees it.
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
                styles[name] = ColwiseParallel(
                    output_layouts=Shard(-1),
                    use_local_output=not self.keep_placements)
            elif role == "row":
                # Always local: by the time this returns, the all-reduce has
                # already made the value Replicate, so to_local() loses nothing,
                # and whatever consumes it is back in the unsharded residual
                # stream.  See the class docstring above for the mixed-tensor
                # error that made this the rule.
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
