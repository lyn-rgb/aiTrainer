"""Capability-probed kernel interfaces with eager PyTorch implementations."""

from .backend import KernelBackend, KernelCapability, KernelSelection, KernelSelectionError, KernelStatus
from .attention import attention_capability, flash_attention_available, scaled_dot_product_attention
from .fused import (fused_adamw_step, fused_mlp, layer_norm, residual_add, rms_norm)

__all__ = ["KernelBackend", "KernelCapability", "KernelSelection", "KernelSelectionError", "KernelStatus",
           "attention_capability", "flash_attention_available", "scaled_dot_product_attention",
           "fused_adamw_step", "fused_mlp", "layer_norm", "residual_add", "rms_norm"]
