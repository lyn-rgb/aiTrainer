"""Fused-operation interfaces backed by numerically transparent eager ops."""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for kernel execution") from exc


def residual_add(value: Any, residual: Any, *, alpha: float = 1.0) -> Any:
    if tuple(value.shape) != tuple(residual.shape):
        raise ValueError("residual and value shapes must match")
    return value + residual * alpha


def rms_norm(value: Any, weight: Any | None = None, *, eps: float = 1e-6) -> Any:
    torch = _torch()
    if value.ndim < 1 or eps <= 0:
        raise ValueError("rms_norm requires rank >= 1 and eps > 0")
    variance = value.float().pow(2).mean(dim=-1, keepdim=True)
    # Multiply in float32 and cast the PRODUCT.  Casting the rsqrt factor down
    # first rounded twice (and rounded a small factor into a coarse grid), giving
    # bf16 inputs a ~7e-3 relative deviation against a declared budget of 0.0.
    result = (value.float() * torch.rsqrt(variance + eps)).to(dtype=value.dtype)
    return result if weight is None else result * weight


def layer_norm(value: Any, weight: Any | None = None, bias: Any | None = None,
               *, eps: float = 1e-5) -> Any:
    torch = _torch()
    if value.ndim < 1 or eps <= 0:
        raise ValueError("layer_norm requires rank >= 1 and eps > 0")
    normalized_shape = (value.shape[-1],)
    return torch.nn.functional.layer_norm(value, normalized_shape, weight, bias, eps)


def fused_mlp(value: Any, weight_in: Any, weight_out: Any, *, bias_in: Any | None = None,
              bias_out: Any | None = None, activation: str = "gelu") -> Any:
    torch = _torch()
    hidden = torch.nn.functional.linear(value, weight_in, bias_in)
    if activation == "gelu":
        hidden = torch.nn.functional.gelu(hidden)
    elif activation == "silu":
        hidden = torch.nn.functional.silu(hidden)
    elif activation == "relu":
        hidden = torch.relu(hidden)
    else:
        raise ValueError(f"unsupported MLP activation {activation!r}")
    return torch.nn.functional.linear(hidden, weight_out, bias_out)


def fused_adamw_step(parameter: Any, gradient: Any, exp_avg: Any, exp_avg_sq: Any, *,
                     step: int, lr: float, betas: tuple[float, float] = (0.9, 0.999),
                     eps: float = 1e-8, weight_decay: float = 0.0) -> int:
    """Reference AdamW update used by fused backends and numerical comparisons."""
    if step < 0 or lr < 0 or eps <= 0 or weight_decay < 0:
        raise ValueError("invalid AdamW hyperparameters")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError("AdamW betas must be in [0, 1)")
    if parameter.shape != gradient.shape or parameter.shape != exp_avg.shape or parameter.shape != exp_avg_sq.shape:
        raise ValueError("AdamW parameter, gradient and state shapes must match")
    with _torch().no_grad():
        next_step = step + 1
        beta1, beta2 = betas
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** next_step
        bias_correction2 = 1.0 - beta2 ** next_step
        denom = exp_avg_sq.sqrt() / bias_correction2 ** 0.5
        denom.add_(eps)
        parameter.mul_(1.0 - lr * weight_decay)
        parameter.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
    return next_step
