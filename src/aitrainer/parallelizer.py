"""Deterministic PP -> TP/SP -> FSDP composition."""

from __future__ import annotations

from typing import Any

from .capability import validate_capabilities
from .config import FrameworkConfig
from .parallel.fsdp import wrap_fsdp
from .parallel.tp import parallelize_tensor_parallel, sequence_parallel_styles
from .mesh import DeviceMeshManager
from .plugins.transformer import TransformerTPPlan
from .parallel.pp_shapes import split_sequential
from .distributed_model import PipelineStage


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
    device_type = str(getattr(getattr(runtime, "state", None), "device", "cpu")).split(":", 1)[0]
    mesh = mesh or DeviceMeshManager(
        pp_size=config.parallel.pp_size, dp_size=config.parallel.dp_size,
        tp_size=config.parallel.tp_size, world_size=runtime.world_size,
        device_type=device_type,
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
    # TP and SP go through one ``parallelize_module`` call so that a single
    # device mesh and a single style dict describe the stage.  They stay
    # independent otherwise: the plan decides the projections, and SP adds the
    # norms.  See ``plugins.transformer.TransformerTPPlan.styles`` for why SP
    # does not also move the projection layouts.
    sequence_parallel = config.parallel.sp_backend != "none"
    if config.parallel.tp_size > 1 or sequence_parallel:
        plan = tp_plan or TransformerTPPlan()
        styles = plan.styles(stage)
        if sequence_parallel:
            styles.update(sequence_parallel_styles(stage))
        plan.validate_dimensions(stage, tp_size=config.parallel.tp_size)
        stage = parallelize_tensor_parallel(
            stage, styles=styles, mesh=mesh.submesh("tp"),
            process_group=process_group, device_type=device_type)
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
