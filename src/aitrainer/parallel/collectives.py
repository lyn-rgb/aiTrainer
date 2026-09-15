"""Synchronous, autograd-aware tensor-parallel collectives."""

from __future__ import annotations

from typing import Any

from ..lifecycle import AsyncOp, ExecutionScheduler


def _dist_world_size(group: Any = None) -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(group)
    except ImportError:
        return 1
    return 1


def _normalize_dim(dim: int, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        raise ValueError(f"dim={dim} is invalid for tensor rank {ndim}")
    return dim % ndim


def _all_reduce(value: Any, group: Any) -> Any:
    import torch.distributed as dist
    dist.all_reduce(value, group=group)
    return value


def all_reduce_gradient(value: Any, group: Any = None) -> Any:
    """All-reduce a gradient in place when the group spans more than one rank.

    Used for parameters that are REPLICATED across the group but whose gradients
    are computed from a local shard of the data -- without this, each copy drifts.
    """
    if _dist_world_size(group) > 1:
        _all_reduce(value, group)
    return value


def _async_or_sync(name: str, handle: Any, *, scheduler: ExecutionScheduler | None = None,
                   tensor: Any = None, group: Any = None, bytes_: int = 0) -> AsyncOp:
    """Wrap a torch.distributed Work handle without ever dropping it.

    ``handle`` may be ``None`` for CPU/Gloo or an unavailable async backend;
    the returned operation still follows the same lifecycle and can be drained.
    """
    op = AsyncOp(name=name, handle=handle, tensor=tensor, group=group, bytes=bytes_).submit()
    target = scheduler or ExecutionScheduler()
    target.register(op)
    return op


def all_reduce_async(value: Any, group: Any = None, *, scheduler: ExecutionScheduler | None = None,
                     op: Any = None) -> AsyncOp:
    import torch.distributed as dist
    if _dist_world_size(group) == 1:
        return _async_or_sync("all_reduce", None, scheduler=scheduler, tensor=value, group=group,
                              bytes_=int(getattr(value, "numel", lambda: 0)()) * int(getattr(value, "element_size", lambda: 1)()))
    kwargs = {"group": group, "async_op": True}
    if op is not None: kwargs["op"] = op
    work = dist.all_reduce(value, **kwargs)
    return _async_or_sync("all_reduce", work, scheduler=scheduler, tensor=value, group=group,
                          bytes_=int(value.numel()) * int(value.element_size()))


def all_gather_async(outputs: list[Any], value: Any, group: Any = None, *,
                     scheduler: ExecutionScheduler | None = None) -> AsyncOp:
    import torch.distributed as dist
    if _dist_world_size(group) == 1:
        outputs[0].copy_(value)
        return _async_or_sync("all_gather", None, scheduler=scheduler, tensor=value, group=group,
                              bytes_=int(value.numel()) * int(value.element_size()))
    work = dist.all_gather(outputs, value.contiguous(), group=group, async_op=True)
    return _async_or_sync("all_gather", work, scheduler=scheduler, tensor=value, group=group,
                          bytes_=sum(int(x.numel()) * int(x.element_size()) for x in outputs))


def reduce_scatter_async(output: Any, inputs: list[Any], group: Any = None, *,
                         scheduler: ExecutionScheduler | None = None) -> AsyncOp:
    import torch.distributed as dist
    if _dist_world_size(group) == 1:
        output.copy_(inputs[0])
        return _async_or_sync("reduce_scatter", None, scheduler=scheduler, tensor=output, group=group,
                              bytes_=int(output.numel()) * int(output.element_size()))
    work = dist.reduce_scatter(output, inputs, group=group, async_op=True)
    return _async_or_sync("reduce_scatter", work, scheduler=scheduler, tensor=output, group=group,
                          bytes_=int(output.numel()) * int(output.element_size()) * len(inputs))


class _CopyToTP:
    @staticmethod
    def apply(value: Any, group: Any) -> Any:
        import torch
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: Any) -> Any:
                ctx.group = group
                return x

            @staticmethod
            def backward(ctx: Any, grad: Any) -> tuple[Any, None]:
                if _dist_world_size(ctx.group) > 1:
                    grad = grad.contiguous().clone()
                    _all_reduce(grad, ctx.group)
                return grad, None
        return Function.apply(value)


class _ReduceFromTP:
    @staticmethod
    def apply(value: Any, group: Any, reduce_dtype: Any | None) -> Any:
        import torch
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: Any, reduce_dtype: Any) -> Any:
                ctx.group, ctx.input_dtype = group, x.dtype
                if _dist_world_size(group) > 1:
                    reduced = x.contiguous().clone()
                    if reduce_dtype is not None:
                        acc_dtype = reduce_dtype
                        if isinstance(acc_dtype, str):
                            name = acc_dtype.replace("torch.", "")
                            if not hasattr(torch, name):
                                raise ValueError(f"unsupported reduce dtype {acc_dtype!r}")
                            acc_dtype = getattr(torch, name)
                        reduced = reduced.to(dtype=acc_dtype)
                    _all_reduce(reduced, group)
                    x = reduced.to(dtype=ctx.input_dtype) if reduced.dtype != ctx.input_dtype else reduced
                return x

            @staticmethod
            def backward(ctx: Any, grad: Any) -> tuple[Any, None]:
                return grad, None
        return Function.apply(value, reduce_dtype)


class _Scatter:
    @staticmethod
    def apply(value: Any, dim: int, group: Any) -> Any:
        import torch
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: Any, dim: int, group: Any) -> Any:
                ctx.dim, ctx.group = dim, group
                if _dist_world_size(group) == 1:
                    return x
                d = _normalize_dim(dim, x.ndim)
                size = x.shape[d]
                world = _dist_world_size(group)
                if size % world:
                    raise ValueError(f"dimension {size} is not divisible by TP world size {world}")
                chunks = x.chunk(world, dim=d)
                import torch.distributed as dist
                rank = dist.get_rank(group)
                return chunks[rank].contiguous()

            @staticmethod
            def backward(ctx: Any, grad: Any) -> tuple[Any, None, None]:
                return _gather_impl(grad, ctx.dim, ctx.group), None, None
        return Function.apply(value, dim, group)


class _Gather:
    @staticmethod
    def apply(value: Any, dim: int, group: Any) -> Any:
        import torch
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: Any, dim: int, group: Any) -> Any:
                ctx.dim, ctx.group = dim, group
                return _gather_impl(x, dim, group)

            @staticmethod
            def backward(ctx: Any, grad: Any) -> tuple[Any, None, None]:
                return _scatter_impl(grad, ctx.dim, ctx.group), None, None
        return Function.apply(value, dim, group)


def _gather_impl(value: Any, dim: int, group: Any) -> Any:
    world = _dist_world_size(group)
    if world == 1:
        return value
    import torch
    d = _normalize_dim(dim, value.ndim)
    outputs = [torch.empty_like(value) for _ in range(world)]
    import torch.distributed as dist
    dist.all_gather(outputs, value.contiguous(), group=group)
    return torch.cat(outputs, dim=d).contiguous()


def _scatter_impl(value: Any, dim: int, group: Any) -> Any:
    world = _dist_world_size(group)
    if world == 1:
        return value
    d = _normalize_dim(dim, value.ndim)
    size = value.shape[d]
    if size % world:
        raise ValueError(f"dimension {size} is not divisible by world size {world}")
    import torch.distributed as dist
    rank = dist.get_rank(group)
    return value.chunk(world, dim=d)[rank].contiguous()


def copy_to_tp(value: Any, group: Any = None) -> Any:
    return _CopyToTP.apply(value, group)


def reduce_from_tp(value: Any, group: Any = None, *, reduce_dtype: Any | None = None) -> Any:
    """All-reduce in an optional accumulation dtype, returning input dtype."""
    return _ReduceFromTP.apply(value, group, reduce_dtype)


def scatter_to_tp(value: Any, group: Any = None, *, dim: int = -1) -> Any:
    return _Scatter.apply(value, dim, group)


def gather_from_tp(value: Any, group: Any = None, *, dim: int = -1) -> Any:
    return _Gather.apply(value, dim, group)


def scatter_to_sequence(value: Any, group: Any = None, *, dim: int = 1) -> Any:
    return scatter_to_tp(value, group, dim=dim)


def gather_from_sequence(value: Any, group: Any = None, *, dim: int = 1) -> Any:
    return gather_from_tp(value, group, dim=dim)


def all_to_all_layout(value: Any, *, scatter_dim: int, gather_dim: int, group: Any = None) -> Any:
    """Synchronous all-to-all for equal contiguous chunks on two dimensions."""
    import torch
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.scatter_dim, ctx.gather_dim, ctx.group = scatter_dim, gather_dim, group
            return _all_to_all_impl(x, scatter_dim, gather_dim, group)

        @staticmethod
        def backward(ctx: Any, grad: Any) -> tuple[Any]:
            value = _all_to_all_impl(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group)
            return (value,)
    return Function.apply(value)


def _all_to_all_impl(value: Any, scatter_dim: int, gather_dim: int, group: Any) -> Any:
    world = _dist_world_size(group)
    if world == 1:
        return value
    import torch
    sd, gd = _normalize_dim(scatter_dim, value.ndim), _normalize_dim(gather_dim, value.ndim)
    if value.shape[sd] % world:
        raise ValueError(f"scatter dimension {value.shape[sd]} is not divisible by world size {world}")
    chunks = value.chunk(world, dim=sd)
    import torch.distributed as dist
    outputs = [torch.empty_like(chunks[0]) for _ in range(world)]
    dist.all_to_all(outputs, list(chunks), group=group)
    return torch.cat(outputs, dim=gd).contiguous()
