"""Scaled dot-product attention with capability-probed eager fallback."""

from __future__ import annotations

from typing import Any


def flash_attention_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and hasattr(torch.nn.functional, "scaled_dot_product_attention") and
                    hasattr(torch.backends, "cuda") and torch.backends.cuda.flash_sdp_enabled())
    except ImportError:
        return False


def _eager_attention(query: Any, key: Any, value: Any, mask: Any | None,
                     dropout_p: float, is_causal: bool) -> Any:
    import torch
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
    return torch.matmul(weights, value)


def scaled_dot_product_attention(query: Any, key: Any, value: Any, *, mask: Any | None = None,
                                 dropout_p: float = 0.0, is_causal: bool = False,
                                 backend: str | None = None) -> Any:
    """Run attention using SDPA when explicitly available, otherwise eager math."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects [batch, heads, sequence, head_dim] tensors")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("query/key/value head and sequence dimensions are incompatible")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if backend not in (None, "eager", "sdpa", "flash"):
        raise ValueError(f"unknown attention backend {backend!r}")
    try:
        import torch
        use_sdpa = backend in (None, "sdpa", "flash") and hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if backend == "flash" and not flash_attention_available():
            raise RuntimeError("flash attention was explicitly requested but is unavailable")
        if backend == "flash":
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                                    enable_mem_efficient=False):
                    return torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError("flash attention execution failed without an eager fallback") from exc
        if use_sdpa:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)
        return _eager_attention(query, key, value, mask, dropout_p, is_causal)
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for attention kernels") from exc
