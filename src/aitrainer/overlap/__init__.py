"""Advanced overlap runtime with explicit synchronization and safe fallback."""
from ..lifecycle import AsyncOp, AsyncState, ExecutionScheduler, LifecycleError, drain
from .controller import OverlapController, OverlapRecord
from .gradient import GradientBucket, GradientBucketReducer
from .buffers import BufferLease, PingPongBuffer
from .backpressure import Backpressure, ResourceBudget
from .parameter import ParameterPrefetchCoordinator, TraceEntry, TraceFingerprint
from .transfer import TransferScheduler
from .metrics import OverlapMetrics
from .microbatch import MicrobatchInterleaveScheduler, MicrobatchResult
__all__ = ["AsyncOp", "AsyncState", "ExecutionScheduler", "LifecycleError", "drain", "OverlapController", "OverlapRecord", "GradientBucket", "GradientBucketReducer", "BufferLease", "PingPongBuffer", "Backpressure", "ResourceBudget", "ParameterPrefetchCoordinator", "TraceEntry", "TraceFingerprint", "TransferScheduler", "OverlapMetrics", "MicrobatchInterleaveScheduler", "MicrobatchResult"]
