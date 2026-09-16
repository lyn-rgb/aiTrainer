"""The curated public surface.

This is the only place the framework's public names are enumerated.  It is a
hand-written list rather than a wildcard re-export, so the boundary is a decision
rather than an accident: adding something to a subsystem does not silently
publish it.

Names are grouped by the layer they come from (see ``docs/架构.md``) -- the
import block above is laid out in layer order, foundations through interfaces,
and each import names its source module.

``__all__`` itself is alphabetical rather than grouped, because ruff's RUF022
requires it to be sorted and a layer-grouped list is not.  Ordering is worth
less here than the linter rule; the imports above carry the grouping.
"""

from .adapters import DataProvider, LossFn, ModelAdapter, OptimizerFactory
from .capability import (
                         Capability,
                         CapabilityStatus,
                         UnsupportedCombinationError,
                         capability_matrix,
                         capability_status,
                         validate_capabilities,
)
from .checkpoint import (
                         CheckpointConverter,
                         CheckpointError,
                         CheckpointManager,
                         CheckpointSchemaError,
                         ConversionError,
                         IncompleteCheckpointError,
)
from .config import (
                         AdvancedOverlapConfig,
                         CompileConfig,
                         ConfigurationError,
                         FrameworkConfig,
                         FSDPConfig,
                         MixedPrecisionConfig,
                         OffloadConfig,
                         ParallelConfig,
                         PlanningConfig,
                         PrecisionConfig,
)
from .core.lifecycle import AsyncOp, ExecutionScheduler
from .data import BatchContract, BatchMetadata, DataContractError, StatefulSampler
from .diagnostics import diagnose_exception, dry_run, inspect_checkpoint, inspect_topology, validate
from .distributed_model import PipelineStage
from .kernels import (
                         KernelBackend,
                         KernelCapability,
                         KernelSelection,
                         KernelSelectionError,
                         KernelStatus,
                         attention_capability,
                         flash_attention_available,
                         fused_adamw_step,
                         fused_mlp,
                         layer_norm,
                         residual_add,
                         rms_norm,
                         scaled_dot_product_attention,
)
from .loading import Manifest, ModelLoadConfig, ModelLoader, ShardSpec
from .memory import BufferKey, BufferPoolError, PinnedBufferPool
from .mesh import DeviceMeshManager
from .metrics import BenchmarkArtifact
from .offload import (
                         ActivationOffloader,
                         ActivationOffloadError,
                         CPUOptimizerStateOffloader,
                         OffloadError,
                         OffloadManager,
                         OptimizerOffloadError,
                         ParameterOffloader,
                         ParameterOffloadError,
)
from .overlap import (
                         AsyncState,
                         Backpressure,
                         BufferLease,
                         GradientBucket,
                         GradientBucketReducer,
                         MicrobatchInterleaveScheduler,
                         MicrobatchResult,
                         OverlapController,
                         OverlapMetrics,
                         OverlapRecord,
                         ParameterPrefetchCoordinator,
                         PingPongBuffer,
                         ResourceBudget,
                         TransferScheduler,
)
from .parallel.fsdp import clip_grad_norm_, no_gradient_sync, supports_gradient_sync, wrap_fsdp
from .parallel.pp_p2p import P2PCommunicator, P2PError
from .parallel.pp_schedule import (
                         GPipeSchedule,
                         OneFOneBSchedule,
                         PipelineScheduleError,
                         ScheduleOutput,
)
from .parallel.pp_shapes import (
                         PipelineShapeError,
                         StagePlan,
                         TensorSpec,
                         normalize_loss,
                         plan_stages,
                         split_microbatches,
                         split_sequential,
)
from .parallel.sp_ulysses import UlyssesAttention, all_to_all_layout, distributed_attention
from .parallel.tp import (
                         TPConfigurationError,
                         device_mesh,
                         parallelize_tensor_parallel,
                         sequence_parallel_styles,
)
from .parallelizer import parallelize
from .planner import PlanCandidate, PlanningError, PlanReport, apply_candidate, suggest_plan
from .plugins.models import (
                         BERTAdapter,
                         GPTAdapter,
                         LlamaAdapter,
                         TinyTransformerAdapter,
                         TransformerAdapter,
                         TransformerModelConfig,
)
from .plugins.transformer import TransformerTPPlan
from .precision import (
                         PrecisionError,
                         autocast_context,
                         cast_gradients,
                         cast_optimizer_state,
                         resolve_dtype,
                         validate_precision,
)
from .presets import ConfigPreset
from .profiler import Profiler
from .runtime import Runtime, RuntimeState
from .topology import RankCoordinate, RankMapping, Topology, inspect_hardware, make_rank_mapping
from .trainer import StepOutput, Trainer

__all__ = [
                         "ActivationOffloadError",
                         "ActivationOffloader",
                         "AdvancedOverlapConfig",
                         "AsyncOp",
                         "AsyncState",
                         "BERTAdapter",
                         "Backpressure",
                         "BatchContract",
                         "BatchMetadata",
                         "BenchmarkArtifact",
                         "BufferKey",
                         "BufferLease",
                         "BufferPoolError",
                         "CPUOptimizerStateOffloader",
                         "Capability",
                         "CapabilityStatus",
                         "CheckpointConverter",
                         "CheckpointError",
                         "CheckpointManager",
                         "CheckpointSchemaError",
                         "CompileConfig",
                         "ConfigPreset",
                         "ConfigurationError",
                         "ConversionError",
                         "DataContractError",
                         "DataProvider",
                         "DeviceMeshManager",
                         "ExecutionScheduler",
                         "FSDPConfig",
                         "FrameworkConfig",
                         "GPTAdapter",
                         "GPipeSchedule",
                         "GradientBucket",
                         "GradientBucketReducer",
                         "IncompleteCheckpointError",
                         "KernelBackend",
                         "KernelCapability",
                         "KernelSelection",
                         "KernelSelectionError",
                         "KernelStatus",
                         "LlamaAdapter",
                         "LossFn",
                         "Manifest",
                         "MicrobatchInterleaveScheduler",
                         "MicrobatchResult",
                         "MixedPrecisionConfig",
                         "ModelAdapter",
                         "ModelLoadConfig",
                         "ModelLoader",
                         "OffloadConfig",
                         "OffloadError",
                         "OffloadManager",
                         "OneFOneBSchedule",
                         "OptimizerFactory",
                         "OptimizerOffloadError",
                         "OverlapController",
                         "OverlapMetrics",
                         "OverlapRecord",
                         "P2PCommunicator",
                         "P2PError",
                         "ParallelConfig",
                         "ParameterOffloadError",
                         "ParameterOffloader",
                         "ParameterPrefetchCoordinator",
                         "PingPongBuffer",
                         "PinnedBufferPool",
                         "PipelineScheduleError",
                         "PipelineShapeError",
                         "PipelineStage",
                         "PlanCandidate",
                         "PlanReport",
                         "PlanningConfig",
                         "PlanningError",
                         "PrecisionConfig",
                         "PrecisionError",
                         "Profiler",
                         "RankCoordinate",
                         "RankMapping",
                         "ResourceBudget",
                         "Runtime",
                         "RuntimeState",
                         "ScheduleOutput",
                         "ShardSpec",
                         "StagePlan",
                         "StatefulSampler",
                         "StepOutput",
                         "TPConfigurationError",
                         "TensorSpec",
                         "TinyTransformerAdapter",
                         "Topology",
                         "Trainer",
                         "TransferScheduler",
                         "TransformerAdapter",
                         "TransformerModelConfig",
                         "TransformerTPPlan",
                         "UlyssesAttention",
                         "UnsupportedCombinationError",
                         "all_to_all_layout",
                         "apply_candidate",
                         "attention_capability",
                         "autocast_context",
                         "capability_matrix",
                         "capability_status",
                         "cast_gradients",
                         "cast_optimizer_state",
                         "clip_grad_norm_",
                         "device_mesh",
                         "diagnose_exception",
                         "distributed_attention",
                         "dry_run",
                         "flash_attention_available",
                         "fused_adamw_step",
                         "fused_mlp",
                         "inspect_checkpoint",
                         "inspect_hardware",
                         "inspect_topology",
                         "layer_norm",
                         "make_rank_mapping",
                         "no_gradient_sync",
                         "normalize_loss",
                         "parallelize",
                         "parallelize_tensor_parallel",
                         "plan_stages",
                         "residual_add",
                         "resolve_dtype",
                         "rms_norm",
                         "scaled_dot_product_attention",
                         "sequence_parallel_styles",
                         "split_microbatches",
                         "split_sequential",
                         "suggest_plan",
                         "supports_gradient_sync",
                         "validate",
                         "validate_capabilities",
                         "validate_precision",
                         "wrap_fsdp",
]
