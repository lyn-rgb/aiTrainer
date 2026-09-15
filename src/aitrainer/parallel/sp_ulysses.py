"""Ulysses-style attention exchange with explicit head/sequence layouts."""

from __future__ import annotations

from typing import Any

from .collectives import all_to_all_layout


class SPConfigurationError(ValueError):
    """Raised for invalid sequence/head partition shapes."""


def _world_size(group: Any) -> int:
    try:
        import torch.distributed as dist
        return dist.get_world_size(group) if dist.is_initialized() else 1
    except ImportError:
        return 1


def pad_sequence(value: Any, seq_lens: Any, *, multiple: int, dim: int = 1,
                 pad_value: float = 0.0) -> tuple[Any, Any]:
    """Pad a tensor to a length divisible by ``multiple``.

    ``value`` must already be the GLOBAL sequence: the pad target is derived from
    the tensor's own length.  Using ``max(seq_lens)`` instead mixed units -- a
    global length applied to this rank's local slice -- which padded the slice
    ``multiple`` times too long and made the post-exchange key axis ``multiple**2``
    times the local length.  Callers that hold a shard must pad the global
    sequence before sharding.
    """
    import torch
    if value.ndim < 2:
        raise SPConfigurationError("sequence padding expects a batched tensor")
    d = dim % value.ndim
    lengths = torch.as_tensor(seq_lens, device=value.device, dtype=torch.long)
    if lengths.ndim != 1 or lengths.numel() != value.shape[0]:
        raise SPConfigurationError("seq_lens must contain one length per batch item")
    if bool((lengths < 1).any()):
        raise SPConfigurationError("seq_lens must be positive")
    target = int(((int(value.shape[d]) + multiple - 1) // multiple) * multiple)
    if target > value.shape[d]:
        shape = list(value.shape)
        shape[d] = target - value.shape[d]
        value = torch.cat((value, torch.full(shape, pad_value, dtype=value.dtype, device=value.device)), dim=d)
    return value, lengths


def unpad_sequence(value: Any, seq_lens: Any, *, dim: int = 1) -> list[Any]:
    """Return per-example tensors trimmed to their valid sequence lengths."""
    d = dim % value.ndim
    if d == 0:
        raise SPConfigurationError("unpad_sequence expects a non-batch sequence dimension")
    return [value[index].narrow(d - 1, 0, int(length)) for index, length in enumerate(seq_lens)]


def apply_rope(value: Any, cos: Any, sin: Any, *, position_offset: int = 0) -> Any:
    """Apply rotary embeddings to ``[B,L,H,D]`` values."""
    import torch
    if value.shape[-1] % 2:
        raise SPConfigurationError("RoPE head_dim must be even")
    length = value.shape[1]
    cos = cos[position_offset:position_offset + length]
    sin = sin[position_offset:position_offset + length]
    if cos.shape[-1] * 2 == value.shape[-1]:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
    if cos.shape[-1] != value.shape[-1] or sin.shape[-1] != value.shape[-1]:
        raise SPConfigurationError("RoPE cos/sin head dimension does not match value")
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1).reshape_as(value)
    return value * cos[None, :, None, :] + rotated * sin[None, :, None, :]


def ulysses_exchange(value: Any, *, group: Any = None, scatter_dim: int,
                     gather_dim: int) -> Any:
    world = _world_size(group)
    if value.shape[scatter_dim] % world:
        raise SPConfigurationError(
            f"dimension {value.shape[scatter_dim]} is not divisible by SP world size {world}")
    return all_to_all_layout(value, scatter_dim=scatter_dim, gather_dim=gather_dim, group=group)


def distributed_attention(q: Any, k: Any, v: Any, *, group: Any = None,
                          attention_mask: Any = None, is_causal: bool = False,
                          scale: float | None = None, seq_lens: Any = None,
                          cos: Any = None, sin: Any = None,
                          position_offset: int = 0) -> Any:
    """Exchange heads for attention, then restore sequence/head layout.

    Inputs use ``[batch, sequence, heads, head_dim]``. Variable lengths are
    padded before communication and returned in padded form; callers can use
    :func:`unpad_sequence` to recover each sample without losing metadata.
    """
    import torch
    world = _world_size(group)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise SPConfigurationError("Ulysses attention expects rank-4 [B,L,H,D] tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise SPConfigurationError("q, k and v must have identical shapes")
    if q.shape[2] % world:
        raise SPConfigurationError(f"head count {q.shape[2]} is not divisible by SP world size {world}")
    lengths = None
    if seq_lens is not None:
        if world > 1:
            # RETAINED AS A LOUD REFUSAL, not silently approximated.  With a sharded
            # sequence the mask has to model WHERE each rank's padding sits after
            # the head/sequence exchange, and this path does not: measured against
            # dense SDPA it returned an attention output with max abs error 0.42 at
            # world=2 (2.4e-7 with seq_lens=None).  Refusing beats training on
            # silently wrong attention.  Pad the GLOBAL sequence before sharding, or
            # use seq_lens=None for dense evenly-divisible batches.
            raise SPConfigurationError(
                "variable-length (seq_lens) Ulysses attention is not implemented for a "
                "sharded sequence: the post-exchange padding layout is not modelled, so "
                "this path previously returned wrong attention at world >= 2. Pass "
                "seq_lens=None, or pad the global sequence before sharding.")
        q, lengths = pad_sequence(q, seq_lens, multiple=world, dim=1)
        k, _ = pad_sequence(k, seq_lens, multiple=world, dim=1)
        v, _ = pad_sequence(v, seq_lens, multiple=world, dim=1)
        if attention_mask is None:
            total = q.shape[1] * world
            positions = torch.arange(total, device=q.device)
            attention_mask = positions[None, None, None, :] < lengths[:, None, None, None]
    if (cos is None) != (sin is None):
        raise SPConfigurationError("cos and sin must be provided together")
    local_q = ulysses_exchange(q, group=group, scatter_dim=2, gather_dim=1)
    local_k = ulysses_exchange(k, group=group, scatter_dim=2, gather_dim=1)
    local_v = ulysses_exchange(v, group=group, scatter_dim=2, gather_dim=1)
    if cos is not None:
        local_q = apply_rope(local_q, cos, sin, position_offset=position_offset)
        local_k = apply_rope(local_k, cos, sin, position_offset=position_offset)
    local_q, local_k, local_v = local_q.transpose(1, 2), local_k.transpose(1, 2), local_v.transpose(1, 2)
    local = torch.nn.functional.scaled_dot_product_attention(
        local_q, local_k, local_v, attn_mask=attention_mask,
        dropout_p=0.0, is_causal=is_causal, scale=scale)
    return ulysses_exchange(local.transpose(1, 2), group=group, scatter_dim=1, gather_dim=2)


class UlyssesAttention:
    """Callable adapter that keeps the communication group explicit."""

    def __init__(self, process_group: Any = None) -> None:
        self.process_group = process_group

    def __call__(self, q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
        return distributed_attention(q, k, v, group=self.process_group, **kwargs)
