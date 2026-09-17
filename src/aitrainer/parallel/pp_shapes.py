"""Pipeline tensor metadata, stage planning, and micro-batch utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.batching import (
    BatchContractError as PipelineShapeError,
)
from ..core.batching import (
    normalize_loss,
    split_microbatches,
)
from ..core.torch import module_base

# `normalize_loss` and `split_microbatches` now live in core.batching; they are
# re-exported here because this module is where callers have always imported
# them from.  The list is load-bearing: without it ruff reads the imports as
# unused and deletes them.
__all__ = ["PipelineSequence", "PipelineShapeError", "StagePlan", "TensorSpec",
           "execution_units", "normalize_loss", "plan_stages", "split_microbatches",
           "split_sequential", "stage_assignment"]

# Must be an expression at import time: the class below inherits from it.
ModuleBase, nn = module_base()


class PipelineSequence(ModuleBase):
    """Run a list of named children in order, **keeping their original names**.

    ``nn.Sequential`` renames its children by position, and this split used to
    build one per stage.  Two consequences followed, neither of them an error:

    * a name-matching TP plan (``q_proj``, ``o_proj``, ...) finds nothing on a
      pipeline stage, so ``pp`` and ``tp`` could not be combined at all.  That
      is now a loud ``TPConfigurationError``, but it is a gap either way;
    * every stage's checkpoint keys start at ``"0"``, so stage 0's ``0.weight``
      and stage 1's ``0.weight`` are *different tensors under one key*.  DCP
      keeps one and hands it to both -- measured at pp=2 as a round trip that
      returned each rank the same tensor, neither one its own, max|dW| = 6.4e-01,
      with save and load both reporting success.

    Both come from the same re-indexing, so fixing it here fixes both.  The
    ordering contract is unchanged: children still run in their original order,
    which is the assumption ``split_sequential`` always carried (a model whose
    ``forward`` is not a plain chain is not splittable by this function, before
    or after).
    """

    def __init__(self, named_layers: Sequence[tuple[str, Any]]) -> None:
        if nn is None:  # type: ignore[truthy-function]
            raise ImportError("PipelineSequence requires PyTorch")
        super().__init__()
        self.layer_names: list[str] = []
        for name, layer in named_layers:
            # Names come from named_children(), so they are non-empty and
            # contain no ".", which is all add_module requires.
            self.add_module(name, layer)
            self.layer_names.append(name)

    def forward(self, value: Any) -> Any:
        for name in self.layer_names:
            value = getattr(self, name)(value)
        return value


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
    def from_tensor(cls, tensor: Any, *, stage: int, microbatch: int, tag: str) -> TensorSpec:
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


def execution_units(module: Any, pp_size: int, *, execution_order: Any = None
                    ) -> list[tuple[str, Any]]:
    """The ordered ``(name, module)`` pairs a pipeline split divides.

    By default that is ``module.named_children()``, which requires the model's
    ``forward`` to be a plain chain of its **direct** children -- the contract
    this function has always carried, and one that a HuggingFace model does not
    meet.  ``LlamaForCausalLM``'s direct children are ``model`` (the whole
    trunk) and ``lm_head``; its four transformer layers are one level further
    down, inside ``model.layers``.  Splitting at the top level therefore puts
    the entire transformer on stage 0, and the parameter counts make that plain
    before anything runs.

    ``execution_order`` is the way out, and it is the caller's to write because
    only they know how their model's ``forward`` is composed.  It receives the
    module and returns the ordered units, including the glue::

        def llama_execution_order(model):
            return [("embed_tokens", model.model.embed_tokens),
                    *[(f"layer_{index}", layer)
                      for index, layer in enumerate(model.model.layers)],
                    ("norm", model.model.norm),
                    ("lm_head", model.lm_head)]

    Names must be dot-free: :class:`PipelineSequence` registers each unit with
    ``add_module``, which rejects ``"."``.  They are the stage's checkpoint keys
    and the prefixes its TP styles are addressed by, so the leaf names are what
    matter -- ``layer_0.attn.q_proj`` still matches ``q_proj`` in the default
    plan.

    With no ``execution_order``, a split that would hand some stage a child
    **which cannot run at all** is refused rather than returned: see
    :func:`_refuse_unrunnable_units`.
    """
    if execution_order is not None:
        units = list(execution_order(module))
        for name, _ in units:
            if not name or "." in name:
                raise PipelineShapeError(
                    f"execution_order returned the unit name {name!r}; names must be "
                    "non-empty and must not contain '.', because PipelineSequence "
                    "registers each unit with nn.Module.add_module, which rejects "
                    "dots. Use one name per unit, e.g. 'layer_0'.")
        return units
    named = list(module.named_children())
    _refuse_unrunnable_units(named, pp_size)
    return named


def _refuse_unrunnable_units(named: Sequence[tuple[str, Any]], pp_size: int) -> None:
    """Refuse a split whose stages are guaranteed to fail at forward time.

    A ``PipelineSequence`` runs each child by calling it, so a child with no
    ``forward`` of its own -- a bare ``nn.Module``, or an ``nn.ModuleList``,
    which is how every HuggingFace model holds its layers -- can never be a
    pipeline unit.  Measured on a Llama-shaped module at ``pp_size=2``: the split
    returned ``['model']`` and ``['lm_head']`` without complaint, and the
    failure only appeared when stage 0 was first called, as

        NotImplementedError: Module [Module] is missing the required "forward"
        function

    -- which names neither the pipeline split nor the module that caused it.
    The check proves something stronger than "this looks wrong": no assignment
    of these children to stages can run.

    Only checked on the default path.  A caller who supplied ``execution_order``
    has already said what the units are.
    """
    if pp_size <= 1 or nn is None:
        return
    base_forward = getattr(nn.Module, "forward", None)
    broken = [name for name, child in named
              if getattr(type(child), "forward", None) is base_forward]
    if not broken:
        return
    raise PipelineShapeError(
        f"pp_size={pp_size} cannot split {broken} -- "
        f"{'these modules' if len(broken) > 1 else 'this module'} defines no forward, "
        "so a pipeline stage holding it could never run. A module holding its "
        "layers in an nn.ModuleList (or an inner module) cannot be split by "
        "named_children(); pass execution_order to say what the stages are. See "
        "parallel.pp_shapes.execution_units for the shape it has to return.")


def split_sequential(module: Any, pp_size: int, *, policy: str = "uniform_layers",
                     sample: Any = None, execution_order: Any = None
                     ) -> tuple[Any, tuple[StagePlan, ...]]:
    """Split a module with an ordered ``named_children()`` contract into stages.

    The split **keeps the original module names** (see :class:`PipelineSequence`),
    which is what lets a name-matching TP plan address a pipeline stage and what
    keeps each stage's checkpoint keys distinct.

    ``execution_order`` overrides what the split divides: see
    :func:`execution_units`, which also documents the model shape the default
    cannot handle.
    """
    units = execution_units(module, pp_size, execution_order=execution_order)
    layers = [layer for _, layer in units]
    plans = plan_stages(layers, pp_size, policy=policy, sample=sample)
    stages = tuple(PipelineSequence(units[p.start:p.stop]) for p in plans)
    return stages, plans


def stage_assignment(model: Any, pp_size: int, *, policy: str = "uniform_layers",
                     sample: Any = None, execution_order: Any = None) -> dict[str, int]:
    """``{parameter name: stage index}`` for the split :func:`split_sequential` makes.

    ``CheckpointConverter.convert`` needs this to write per-stage files, and it
    used to demand the caller supply it by hand -- refusing ``pp_size>1``
    without one, because the mapping is not derivable from a dense state dict.
    It is derivable from the *split*: this runs the same ``split_sequential``
    the training path runs and reports which stage owns each parameter, so the
    two cannot disagree about where a layer went.

    The names are the model's own because ``PipelineSequence`` keeps them (see
    there); with the position-renaming ``nn.Sequential`` every stage reported
    ``0.weight`` and the mapping would have been useless.
    """
    stages, plans = split_sequential(model, pp_size, policy=policy, sample=sample,
                                     execution_order=execution_order)
    assignment: dict[str, int] = {}
    for plan, stage in zip(plans, stages):
        for name, _ in stage.named_parameters():
            assignment[name] = plan.stage
    return assignment
