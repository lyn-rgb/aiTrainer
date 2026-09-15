"""Deterministic PP -> TP/SP -> FSDP composition."""

from __future__ import annotations

from typing import Any

from .capability import validate_capabilities
from .config import FrameworkConfig
from .parallel.fsdp import wrap_fsdp
from .mesh import DeviceMeshManager
from .plugins.transformer import TransformerTPPlan, replace_linear_modules
from .parallel.pp_shapes import split_sequential
from .distributed_model import PipelineStage


def _apply_sequence_parallel(module: Any, *, process_group: Any) -> Any:
    """Replace ordinary LayerNorm leaves with explicit TP-group SP layers."""
    import torch
    from .parallel.sp_sequence import SequenceParallelLayerNorm

    for name, child in list(module.named_children()):
        if isinstance(child, SequenceParallelLayerNorm):
            continue
        if isinstance(child, torch.nn.LayerNorm):
            replacement = SequenceParallelLayerNorm(
                child.normalized_shape, eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                process_group=process_group, input_is_parallel=False, gather_output=True,
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    replacement.norm.weight.copy_(child.weight)
                    replacement.norm.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _apply_sequence_parallel(child, process_group=process_group)
    return module


def parallelize(model: Any, *, config: FrameworkConfig, runtime: Any,
                process_group: Any = None, tp_plan: TransformerTPPlan | None = None,
                mesh: DeviceMeshManager | None = None) -> Any:
    """Apply the validated composition in a deterministic ownership order.

    The returned object owns only the current PP stage.  TP/SP modules are
    built against the TP group and FSDP wraps that stage using only the DP
    group.  Supplying a pre-created mesh is recommended; otherwise one is
    created from the validated configuration and runtime world size.
    """
    validate_capabilities(config, world_size=runtime.world_size)
    if runtime.world_size > 1:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                raise RuntimeError("parallelize requires an initialized process group for world_size>1")
        except ImportError as exc:
            raise RuntimeError("PyTorch distributed is required for multi-rank parallelize") from exc
    mesh = mesh or DeviceMeshManager(
        pp_size=config.parallel.pp_size, dp_size=config.parallel.dp_size,
        tp_size=config.parallel.tp_size, world_size=runtime.world_size,
        device_type=str(getattr(getattr(runtime, "state", None), "device", "cpu")).split(":", 1)[0],
    )
    coordinate = mesh.coordinate
    if config.parallel.pp_size > 1:
        policy = config.planning.stage_policy if config.planning.enabled else "uniform_layers"
        stages = split_sequential(model, config.parallel.pp_size, policy=policy)[0]
        stage_id = coordinate.pp
        stage = stages[stage_id]
    else:
        stage_id = 0
        stage = model
    tp_group = process_group if process_group is not None else mesh.tensor_parallel_group
    if config.parallel.sp_backend != "none":
        stage = _apply_sequence_parallel(stage, process_group=tp_group)
    if config.parallel.tp_size > 1:
        stage = replace_linear_modules(stage, process_group=tp_group, plan=tp_plan,
                                       reduce_dtype=config.precision.reduce_dtype)
    if config.fsdp.enabled:
        stage = wrap_fsdp(stage, runtime=runtime, mesh=mesh, config=config.fsdp)
    if config.parallel.pp_size > 1:
        return PipelineStage(stage, stage_id=stage_id,
                             schedule=config.parallel.pp_schedule,
                             pp_group=mesh.pipeline_parallel_group,
                             pp_ranks=mesh.ranks("pp"),
                             pp_size=config.parallel.pp_size,
                             num_microbatches=config.parallel.num_microbatches,
                             tp_rank=coordinate.tp, dp_rank=coordinate.dp)
    return stage
