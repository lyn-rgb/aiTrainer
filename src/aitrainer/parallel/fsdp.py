"""FSDP2 sharding over the orthogonal DP axis.

FSDP owns only the data-parallel process group.  A stage may already contain
TP-sharded parameters and may be one slice of a PP model; those dimensions must
never be folded into the FSDP group.

This is FSDP2 (``torch.distributed.fsdp.fully_shard``), not the FSDP1 wrapper
class.  Two differences shape the code below:

* ``fully_shard`` is **composable and in place** -- it mutates the module it is
  given and returns that same object.  There is no wrapper to hold, so this
  module no longer defines one; ``wrap_fsdp`` returns the module.
* Parameters become ``DTensor``, and gradient reduction is controlled by
  ``set_requires_gradient_sync`` rather than FSDP1's ``no_sync()`` context
  manager.  :func:`no_gradient_sync` is the replacement, and it is not optional:
  a module that silently lacks both would still train, just with one full
  gradient reduction per microbatch instead of one per accumulation window.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..core.dtypes import dtype_name


class FSDPConfigurationError(ValueError):
    """Raised when an unsupported FSDP combination is requested."""


def _mixed_precision(config: Any) -> Any:
    """FSDP2's ``MixedPrecisionPolicy``, or the default (all fp32) policy."""
    from torch.distributed.fsdp import MixedPrecisionPolicy
    if config is None or not getattr(config, "enabled", False):
        return MixedPrecisionPolicy()
    import torch
    dtype = getattr(torch, dtype_name(config.dtype), config.dtype)
    # No buffer_dtype: FSDP2 has no such knob.  output_dtype is new and left at
    # its default so the forward output keeps the parameter dtype.
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)


def _device_mesh(mesh: Any, dp_group: Any, device_type: str) -> Any:
    """A 1-D DeviceMesh over the DP axis, however the caller described it.

    ``fully_shard`` takes a DeviceMesh, never a ProcessGroup.  ``DeviceMeshManager``
    exposes a real DeviceMesh for the whole ``(pp, dp, tp)`` topology, but only
    when the global ranks are in logical order -- for a permuted mapping it
    deliberately keeps process groups instead (``mesh.py``).  ``from_group``
    covers that case, so both descriptions work.

    A caller that supplies neither gets the whole world, which is FSDP1's
    behaviour and FSDP2's own default when ``mesh`` is omitted -- but the mesh is
    built here rather than left to torch, for two separate reasons.  Torch infers
    the device type from the parameters, and on a CPU-only macOS host that
    inference reaches for the custom 'mps' backend and raises before FSDP ever
    starts.  And ``fully_shard`` with no mesh calls ``init_process_group``
    through env:// rendezvous, so a caller at ``world_size == 1`` with no group
    gets ``environment variable RANK expected`` -- an error that names neither
    FSDP nor the missing process group.  FSDP1 failed there too, so this is not a
    regression, only a legible message instead.
    """
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    if isinstance(dp_group, DeviceMesh):
        return dp_group
    if dp_group is None and isinstance(mesh, DeviceMesh):
        return mesh
    if dp_group is None:
        import torch.distributed as dist
        if not dist.is_initialized():
            raise FSDPConfigurationError(
                "FSDP needs an initialized process group; FSDP2 cannot build a "
                "device mesh without one. Initialize a process group first -- a "
                "single-rank one is enough -- then wrap the module.")
        world_size = dist.get_world_size()
        return DeviceMesh(device_type, torch.arange(world_size, dtype=torch.int))
    return DeviceMesh.from_group(dp_group, device_type)


@contextmanager
def no_gradient_sync(module: Any) -> Iterator[None]:
    """Suppress the reduce-scatter for one microbatch.

    FSDP2 replaces FSDP1's ``no_sync()``.  Without this the only symptom is
    performance and reduction timing -- no exception -- so the absence of the
    FSDP2 method on a module must be visible, not silently ignored: callers that
    need synchronisation should use :func:`supports_gradient_sync` to check.
    """
    setter = getattr(module, "set_requires_gradient_sync", None)
    if setter is None:
        yield
        return
    setter(False)
    try:
        yield
    finally:
        setter(True)


def supports_gradient_sync(module: Any) -> bool:
    """Whether ``module`` can defer its gradient reduction.

    Both generations answer yes: FSDP1 via ``no_sync``, FSDP2 via
    ``set_requires_gradient_sync``, and DDP via ``no_sync``.
    """
    return hasattr(module, "no_sync") or hasattr(module, "set_requires_gradient_sync")


def clip_grad_norm_(module: Any, max_norm: float, norm_type: float = 2.0) -> Any:
    """Clip gradients to a **global** norm, over an FSDP2 or plain module.

    FSDP1 exposed ``module.clip_grad_norm_``; FSDP2 has no such method, and
    ``torch.distributed.fsdp.clip_grad_norm_`` does not exist in torch 2.9.

    ``torch.nn.utils.clip_grad_norm_`` looks like the drop-in, and is not one.
    DTensor *does* register ``linalg.vector_norm``, with a ``_NormPartial``
    placement whose local value is this shard's ``sum |g|^p`` and whose
    ``full_tensor()`` is the sum over the mesh -- so it is the right primitive.
    But ``nn.utils`` stacks the per-parameter results with ``torch.stack``, and
    that stack is not DTensor-aware: it silently yields the shard-local numbers.
    Measured at dp=2 on a 3-layer model, against a true global norm of 0.678:
    rank 0 reported 0.551 and rank 1 reported 0.395.  Each rank would then clip
    by a *different* factor, or skip clipping that was required -- and nothing
    raises.

    Hence the explicit reduction below, and the guard that turns the silent
    degradation into an error if a future torch stops registering the norm for
    DTensor.  This is only safe when every rank holds the same parameters (FSDP2
    and DDP both guarantee that); ranks that disagreed about which parameters
    have gradients would pair a different number of collectives and hang.
    """
    import torch
    grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not grads:
        return torch.zeros(())
    norms: list[Any] = []
    for grad in grads:
        norm = torch.linalg.vector_norm(grad, norm_type)
        if hasattr(grad, "placements") and not hasattr(norm, "placements"):
            raise FSDPConfigurationError(
                "torch.linalg.vector_norm returned a local tensor for a DTensor input, "
                "so the gradient norm would be shard-local and each rank would clip by a "
                "different factor. Refusing to clip silently.")
        norms.append(norm.full_tensor().reshape(()) if hasattr(norm, "full_tensor")
                     else norm.reshape(()))
    total = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    clip_coef = float(max_norm) / (float(total) + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return total


def wrap_fsdp(module: Any, *, runtime: Any, mesh: Any = None, config: Any = None) -> Any:
    """Shard ``module`` over the DP axis and return it (mutated in place)."""
    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import OffloadPolicy, fully_shard
    except ImportError as exc:
        raise FSDPConfigurationError("PyTorch FSDP2 is unavailable") from exc
    world_size = int(getattr(runtime, "world_size", 1))
    if world_size > 1 and not dist.is_initialized():
        raise FSDPConfigurationError("initialize Runtime before wrapping a multi-rank model")

    device_type = str(getattr(getattr(runtime, "state", None), "device", "cpu")).split(":", 1)[0]
    # No CPUOffloadPolicy path: FSDPConfig has no field for it, and the CPU
    # offload this framework does implement is FrameworkConfig.offload (see the
    # fsdp_offload capability entry, which is UNSUPPORTED on purpose).
    offload_policy = OffloadPolicy()
    mp_policy = _mixed_precision(getattr(config, "mixed_precision", None))
    # One path for every world size, including 1.  There used to be a separate
    # branch here that called fully_shard with no mesh at all when there was no
    # group, on the grounds that a single-process run should get the same module
    # type and gradient-sync surface as a multi-rank one -- but the branch could
    # not do that: without a mesh, fully_shard performs an env:// rendezvous and
    # dies on the missing RANK variable.  Building the mesh explicitly is what
    # actually delivers the uniformity the branch was written for.
    dp_mesh = dp_axis_mesh(mesh, device_type)
    fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    return module


def dp_axis_mesh(mesh: Any, device_type: str = "cpu") -> Any:
    """The DeviceMesh FSDP should shard over, given a ``DeviceMeshManager``.

    Prefers the ``dp`` **submesh** over the DP process group.  The two cover the
    same ranks and look interchangeable, and are not: ``DeviceMesh.from_group``
    produces a standalone mesh with no parent, while ``mesh["dp"]`` is a view of
    the same ``(pp, dp, tp)`` parent the TP axis came from.  FSDP2 requires the
    two to share a parent as soon as the module holds tensor-parallel DTensor
    parameters, and refuses otherwise:

        FSDP requires the DP and model parallel TP/EP mesh to have the same
        parent mesh but got:
        DP's global mesh: DeviceMesh((1,)), TP/EP's: DeviceMesh((pp=1,dp=1,tp=2))

    Measured: ``tp_size=2`` with ``fsdp.enabled`` failed at construction for
    exactly this reason while ``dp_size=2`` alone worked.  The question could
    not arise before tensor parallelism became DTensor-based, because there were
    no TP DTensors for ``fully_shard`` to reconcile with -- which is why this
    only broke once Stage B landed.
    """
    from torch.distributed.device_mesh import DeviceMesh
    submesh = getattr(mesh, "submesh", lambda _: None)("dp")
    if isinstance(submesh, DeviceMesh):
        return submesh
    group = submesh if submesh is not None else getattr(mesh, "data_parallel_group", None)
    return _device_mesh(mesh, group, device_type)
