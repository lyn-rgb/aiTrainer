"""FSDP FULL_SHARD wrapper for orthogonal DP/TP/PP meshes.

FSDP owns only the data-parallel process group.  A stage may already contain
TP-sharded parameters and may be one slice of a PP model; those dimensions
must never be folded into the FSDP group.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable


class FSDPConfigurationError(ValueError):
    """Raised when an unsupported FSDP combination is requested."""


def fsdp_available() -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
        return True
    except ImportError:
        return False


def _mixed_precision(config: Any) -> Any:
    if not getattr(config, "enabled", False):
        return None
    from torch.distributed.fsdp import MixedPrecision
    import torch
    dtype = getattr(torch, str(config.dtype).replace("torch.", ""), config.dtype)
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


class FSDPWrapper:
    """Construct and own a FULL_SHARD FSDP module and its state-dict context."""

    def __init__(self, module: Any, *, process_group: Any = None, device_id: Any = None,
                 config: Any = None, auto_wrap_policy: Any = None) -> None:
        try:
            import torch
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        except ImportError as exc:
            raise FSDPConfigurationError("PyTorch FSDP is unavailable") from exc
        if config is not None:
            if getattr(config, "sharding", "FULL_SHARD") != "FULL_SHARD":
                raise FSDPConfigurationError("Batch 1 only supports fsdp.sharding='FULL_SHARD'")
            if getattr(config, "cpu_offload", False):
                raise FSDPConfigurationError(
                    "FSDP native cpu_offload is not wired to OffloadManager; configure explicit CPU offload"
                )
        kwargs: dict[str, Any] = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "process_group": process_group,
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": True,
        }
        if device_id is not None:
            kwargs["device_id"] = device_id
        if config is not None:
            mixed = _mixed_precision(getattr(config, "mixed_precision", None))
            if mixed is not None:
                kwargs["mixed_precision"] = mixed
            kwargs["limit_all_gathers"] = bool(getattr(config, "limit_all_gathers", True))
            kwargs["forward_prefetch"] = bool(getattr(config, "forward_prefetch", False) and
                                               getattr(config, "execution_trace_complete", False))
        self.module = FSDP(module, **kwargs)
        self._fsdp = FSDP
        self._torch = torch

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterable[Any]:
        return self.module.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> "FSDPWrapper":
        self.module.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "FSDPWrapper":
        self.module.to(*args, **kwargs)
        return self

    def eval(self) -> "FSDPWrapper":
        return self.train(False)

    def no_sync(self):
        return self.module.no_sync() if hasattr(self.module, "no_sync") else nullcontext()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state: dict[str, Any], **kwargs: Any) -> Any:
        return self.module.load_state_dict(state, **kwargs)

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> Any:
        """Use FSDP's globally correct sharded gradient norm implementation."""
        return self.module.clip_grad_norm_(max_norm, norm_type)

    def full_state_dict_context(self):
        """Return a context that gathers a full state dict only when explicitly requested."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        state = StateDictType.FULL_STATE_DICT
        # offload_to_cpu=True asks for the gathered tensor to be copied to CPU,
        # which is only meaningful when the parameters are on an accelerator.
        # With CPU parameters it segfaults the process outright -- measured at
        # world_size=2 with Gloo, for both rank0_only settings, while
        # offload_to_cpu=False returns the full tensor on every rank as
        # intended.  A hard crash with no traceback is the worst possible
        # failure mode for a checkpoint path.
        offload = any(parameter.device.type == "cuda"
                      for parameter in self.module.parameters())
        cfg = FullStateDictConfig(offload_to_cpu=offload, rank0_only=False)
        return self._fsdp.state_dict_type(self.module, state, cfg)

    def sharded_state_dict_context(self):
        from torch.distributed.fsdp import StateDictType
        state = StateDictType.SHARDED_STATE_DICT
        return self._fsdp.state_dict_type(self.module, state)


def wrap_fsdp(module: Any, *, runtime: Any, mesh: Any = None, config: Any = None) -> FSDPWrapper:
    """Wrap a TP/SP-transformed stage using only its orthogonal DP group."""
    config = config or type("FSDPConfig", (), {"sharding": "FULL_SHARD"})()
    mapping = getattr(mesh, "mapping", None)
    world_size = int(getattr(runtime, "world_size", 1))
    if world_size > 1:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                raise FSDPConfigurationError("initialize Runtime before wrapping a multi-rank model")
        except ImportError as exc:
            raise FSDPConfigurationError("PyTorch distributed is required for multi-rank FSDP") from exc
    device_id = getattr(runtime, "state", None)
    device_id = getattr(device_id, "device", None)
    process_group = getattr(mesh, "data_parallel_group", None)
    if process_group is None:
        process_group = getattr(mesh, "submesh", lambda _: None)("dp")
    if world_size > 1 and mapping is not None and process_group is None:
        raise FSDPConfigurationError(
            "an initialized DeviceMesh with a concrete DP process group is required for multi-rank FSDP"
        )
    # Forward whatever device the runtime resolved.  FSDP infers one when
    # ``device_id`` is omitted, and on a CPU-only macOS host that inference
    # resolves the custom 'mps' backend and raises "Custom backend 'mps' not
    # implement 'torch.mps.current_device'" -- so FSDP could not initialise at
    # all, even though FSDP itself runs on CPU once the device is named.
    # ``Runtime.resolve_device`` always returns a concrete "cpu"/"cuda:N", so
    # there is no "auto" value to filter out here.
    return FSDPWrapper(module, process_group=process_group,
                       device_id=device_id, config=config)
