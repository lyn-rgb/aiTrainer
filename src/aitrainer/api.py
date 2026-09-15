"""Stable public API for the first delivery batch."""

from .adapters import DataProvider, LossFn, ModelAdapter, OptimizerFactory
from .capability import (Capability, CapabilityStatus, UnsupportedCombinationError,
                         capability_matrix, capability_status, validate_capabilities)
from .config import (CompileConfig, ConfigurationError, FrameworkConfig,
                     FSDPConfig, MixedPrecisionConfig, OffloadConfig, AdvancedOverlapConfig, ParallelConfig, PlanningConfig,
                     PrecisionConfig)
from .diagnostics import dry_run, inspect_topology, validate
from .loading import Manifest, ModelLoadConfig, ModelLoader, ShardSpec
from .presets import ConfigPreset
from .trainer import StepOutput, Trainer
from .runtime import Runtime, RuntimeState
from .mesh import DeviceMeshManager
from .topology import RankCoordinate, RankMapping, Topology, inspect_hardware, make_rank_mapping
from .parallel.fsdp import FSDPWrapper, fsdp_available, wrap_fsdp
from .lifecycle import AsyncOp, ExecutionScheduler
from .overlap import (GradientBucket, OverlapController, OverlapRecord, AsyncState, GradientBucketReducer,
                      PingPongBuffer, BufferLease, Backpressure, ResourceBudget,
                      ParameterPrefetchCoordinator, TransferScheduler,
                      OverlapMetrics)
from .overlap import MicrobatchInterleaveScheduler, MicrobatchResult
from .memory import BufferKey, BufferPoolError, PinnedBufferPool
from .offload import (ActivationOffloader, ActivationOffloadError, CPUOptimizerStateOffloader,
                      OffloadError, OffloadManager, OptimizerOffloadError, ParameterOffloader,
                      ParameterOffloadError)
from .parallel.collectives import (all_to_all_layout, copy_to_tp, gather_from_sequence,
                                    gather_from_tp, reduce_from_tp, scatter_to_sequence,
                                    scatter_to_tp, all_reduce_async, all_gather_async,
                                    reduce_scatter_async)
from .parallel.tp import ColumnParallelLinear, RowParallelLinear
from .parallel.sp_sequence import SequenceParallelLayerNorm, gather_sequence, scatter_sequence
from .parallel.sp_ulysses import UlyssesAttention, distributed_attention
from .parallelizer import parallelize
from .plugins.transformer import TransformerTPPlan
from .distributed_model import PipelineStage
from .parallel.pp_shapes import (PipelineShapeError, StagePlan, TensorSpec, normalize_loss,
                                  plan_stages, split_microbatches, split_sequential)
from .parallel.pp_p2p import P2PCommunicator, P2PError
from .parallel.pp_schedule import (GPipeSchedule, OneFOneBSchedule,
                                    PipelineScheduleError, ScheduleOutput)
from .data import BatchContract, BatchMetadata, DataContractError, StatefulSampler
from .checkpoint import CheckpointError, CheckpointManager, CheckpointSchemaError, IncompleteCheckpointError
from .checkpoint import CheckpointConverter, ConversionError
from .metrics import BenchmarkArtifact
from .profiler import Profiler
from .precision import (PrecisionError, autocast_context, cast_gradients, cast_optimizer_state,
                        resolve_dtype, validate_precision)
from .kernels import (KernelBackend, KernelCapability, KernelSelection, KernelSelectionError, KernelStatus,
                      attention_capability, flash_attention_available,
                      scaled_dot_product_attention, fused_adamw_step, fused_mlp,
                      layer_norm, residual_add, rms_norm)
from .planner import PlanCandidate, PlanReport, PlanningError, apply_candidate, suggest_plan
from .plugins.models import (BERTAdapter, GPTAdapter, LlamaAdapter, TinyTransformerAdapter,
                              TransformerAdapter, TransformerModelConfig)
from .diagnostics import diagnose_exception, inspect_checkpoint

__all__ = ["Capability", "CapabilityStatus", "CompileConfig", "ConfigPreset",
           "ConfigurationError", "DataProvider", "FrameworkConfig", "LossFn", "ModelAdapter",
           "FSDPConfig", "Manifest", "MixedPrecisionConfig", "OffloadConfig", "AdvancedOverlapConfig", "PlanningConfig",
           "ModelLoadConfig", "ModelLoader",
           "OptimizerFactory", "ParallelConfig", "PrecisionConfig", "RankCoordinate", "RankMapping",
           "Runtime", "RuntimeState",
           "ShardSpec", "StepOutput", "Topology", "Trainer", "DeviceMeshManager", "FSDPWrapper",
           "fsdp_available", "inspect_hardware", "make_rank_mapping", "wrap_fsdp",
           "AsyncOp", "ExecutionScheduler", "GradientBucket",
           "OverlapController", "OverlapRecord",
           "AsyncState", "GradientBucketReducer", "PingPongBuffer", "BufferLease", "Backpressure", "ResourceBudget",
           "ParameterPrefetchCoordinator", "TransferScheduler", "OverlapMetrics",
           "MicrobatchInterleaveScheduler", "MicrobatchResult",
           "BufferKey", "BufferPoolError", "PinnedBufferPool", "ActivationOffloader",
           "ActivationOffloadError", "CPUOptimizerStateOffloader", "OffloadError", "OffloadManager",
           "OptimizerOffloadError", "ParameterOffloader", "ParameterOffloadError",
           "ColumnParallelLinear", "RowParallelLinear", "copy_to_tp", "reduce_from_tp",
           "scatter_to_tp", "gather_from_tp", "scatter_to_sequence", "gather_from_sequence",
           "all_to_all_layout",
           "all_reduce_async", "all_gather_async", "reduce_scatter_async",
           "parallelize", "TransformerTPPlan",
           "PipelineStage", "TensorSpec", "StagePlan", "PipelineShapeError", "plan_stages",
           "split_microbatches", "split_sequential", "normalize_loss", "P2PCommunicator", "P2PError",
           "GPipeSchedule", "OneFOneBSchedule", "PipelineScheduleError", "ScheduleOutput",
           "BatchContract", "BatchMetadata", "DataContractError", "StatefulSampler",
           "CheckpointError", "CheckpointManager", "CheckpointSchemaError", "IncompleteCheckpointError",
           "CheckpointConverter", "ConversionError",
           "BenchmarkArtifact", "Profiler", "diagnose_exception",
           "inspect_checkpoint",
           "PrecisionError", "autocast_context", "cast_gradients", "cast_optimizer_state",
           "resolve_dtype", "validate_precision",
           "KernelBackend", "KernelCapability", "KernelSelection", "KernelSelectionError", "KernelStatus",
           "attention_capability", "flash_attention_available", "scaled_dot_product_attention",
           "fused_adamw_step", "fused_mlp", "layer_norm", "residual_add", "rms_norm",
           "PlanCandidate", "PlanReport", "PlanningError", "apply_candidate", "suggest_plan",
           "BERTAdapter", "GPTAdapter", "LlamaAdapter", "TinyTransformerAdapter",
           "TransformerAdapter", "TransformerModelConfig",
           "SequenceParallelLayerNorm", "UlyssesAttention", "distributed_attention",
           "scatter_sequence", "gather_sequence",
           "UnsupportedCombinationError", "capability_matrix", "capability_status",
           "validate_capabilities", "dry_run", "inspect_topology", "validate"]
