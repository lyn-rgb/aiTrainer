"""Validated distributed parallel backends."""

from .fsdp import clip_grad_norm_, no_gradient_sync, supports_gradient_sync, wrap_fsdp
from .groups import ProcessGroups
from .pp_shapes import stage_assignment
from .sp_ulysses import UlyssesAttention, all_to_all_layout, distributed_attention
from .tp import (
                 TPConfigurationError,
                 device_mesh,
                 parallelize_tensor_parallel,
                 sequence_parallel_styles,
)

__all__ = ["ProcessGroups", "TPConfigurationError", "UlyssesAttention", "all_to_all_layout",
           "clip_grad_norm_", "device_mesh", "distributed_attention", "no_gradient_sync",
           "parallelize_tensor_parallel", "sequence_parallel_styles",
           "stage_assignment", "supports_gradient_sync", "wrap_fsdp"]
