"""Capability-probed kernel interfaces with eager PyTorch implementations."""

from .attention import flash_attention_available, scaled_dot_product_attention
from .fused import (fused_adamw_step, fused_mlp, layer_norm, residual_add, rms_norm)

__all__ = ["flash_attention_available", "scaled_dot_product_attention",
           "fused_adamw_step", "fused_mlp", "layer_norm", "residual_add", "rms_norm"]
