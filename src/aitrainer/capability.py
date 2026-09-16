"""Capability levels and conservative combination checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util

from .config import ConfigurationError, FrameworkConfig


class CapabilityStatus(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class UnsupportedCombinationError(ConfigurationError):
    """Raised when a requested capability is outside the current stable boundary."""


@dataclass(frozen=True)
class Capability:
    name: str
    status: CapabilityStatus
    reason: str
    fallback: str | None = None


def capability_matrix() -> tuple[Capability, ...]:
    """Return the version-independent capability and composition matrix."""
    return (
        Capability("single_process_eager", CapabilityStatus.STABLE, "PyTorch eager single-process training"),
        Capability("fp32", CapabilityStatus.STABLE, "native float32"),
        Capability("bf16", CapabilityStatus.STABLE, "when the selected device supports it"),
        Capability("fp16", CapabilityStatus.STABLE, "AMP with GradScaler"),
        Capability("checkpoint", CapabilityStatus.STABLE, "atomic local checkpoint"),
        Capability("fsdp_full_shard", CapabilityStatus.STABLE,
                   "FSDP2 fully_shard over the explicit DP group; TP/SP/PP axes remain orthogonal"),
        Capability("fsdp_tp", CapabilityStatus.STABLE,
                   "TP-sharded stage sharded in place by FSDP2 over the orthogonal DP group"),
        Capability("fsdp_sp", CapabilityStatus.STABLE,
                   "sequence-parallel activations over the TP group with FSDP2 DP sharding"),
        Capability("fsdp_sp_tp", CapabilityStatus.STABLE,
                   "TP and sequence-parallel projections with FSDP2 over the DP group"),
        Capability("fsdp_sp_tp_pp", CapabilityStatus.STABLE,
                   "PP stage -> TP/SP -> FSDP2 composition with explicit mesh groups"),
        Capability("fsdp_offload", CapabilityStatus.UNSUPPORTED,
                   "FSDP2 CPUOffloadPolicy is not exposed; use FrameworkConfig.offload, "
                   "which is the offload path this framework implements",
                   "fsdp_full_shard"),
        Capability("tp", CapabilityStatus.STABLE, "synchronous Column/Row Parallel Linear"),
        Capability("sp", CapabilityStatus.STABLE,
                   "sequence-parallel activations over the TP group; config selects only "
                   "'megatron', and 'ulysses' is refused rather than accepted-and-ignored. "
                   "The Ulysses head exchange itself is a separate explicit API that needs "
                   "NCCL -- see UlyssesAttention / distributed_attention"),
        Capability("pp_gpipe_1f1b", CapabilityStatus.STABLE,
                   "non-interleaved pipeline schedules with synchronous P2P"),
        Capability("offload_optimizer", CapabilityStatus.STABLE, "CPU optimizer/state lifecycle"),
        Capability("offload_parameter", CapabilityStatus.EXPERIMENTAL,
                   "synchronous CPU master fetch/release only; ParameterOffloader "
                   "exposes an async prefetch entry point but it stays inert unless a "
                   "trace is finalized, and Trainer only calls fetch/release"),
        Capability("offload_activation", CapabilityStatus.EXPERIMENTAL,
                   "saved_tensors_hooks with explicit context and CPU staging"),
        Capability("offload_nvme", CapabilityStatus.UNSUPPORTED, "NVMe/AIO is outside the stable boundary"),
        Capability("offload", CapabilityStatus.STABLE, "explicit CPU offload modes"),
        Capability("kernel_eager_fallback", CapabilityStatus.EXPERIMENTAL,
                   "KernelBackend exposes eager fallback selection, but no model path "
                   "consults the registry yet -- kernels/attention.py and kernels/fused.py "
                   "run eager directly"),
        Capability("flash_attention", CapabilityStatus.EXPERIMENTAL,
                   "uses PyTorch SDPA/flash dispatch only when probed available", "kernel_eager_fallback"),
        Capability("fused_norm_mlp_residual", CapabilityStatus.EXPERIMENTAL,
                   "numerically transparent eager reference implementation", "kernel_eager_fallback"),
        Capability("fused_adamw", CapabilityStatus.EXPERIMENTAL,
                   "reference update path; no custom CUDA kernel", "kernel_eager_fallback"),
        Capability("overlap_metrics", CapabilityStatus.EXPERIMENTAL,
                   "operation counts, elapsed time and in-flight peaks are real; "
                   "overlapped_seconds has no writer yet, so a controller's "
                   "overlap_ratio is structurally zero"),
        Capability("async_overlap", CapabilityStatus.EXPERIMENTAL,
                   "requires caller-owned validated async handles; disabled by default", "overlap_metrics"),
        Capability("gradient_bucket_overlap", CapabilityStatus.EXPERIMENTAL,
                   "backward-ready buckets for non-FSDP reducers; FSDP combination is rejected", "overlap_metrics"),
        Capability("tp_bulk_overlap", CapabilityStatus.EXPERIMENTAL,
                   "pure PyTorch AG/RS bulk overlap with synchronous CPU/Gloo fallback", "tp"),
        Capability("parameter_prefetch", CapabilityStatus.EXPERIMENTAL,
                   "trace-validated parameter fetch/prefetch with bounded residency", "offload_parameter"),
        Capability("pp_p2p_overlap", CapabilityStatus.EXPERIMENTAL,
                   "the preposted receive API (post_recv_forward/backward, wait_posted) "
                   "has no caller and the 1F1B path issues async sends unconditionally, "
                   "so this switch currently only raises when pp_size<=1", "pp_gpipe_1f1b"),
        Capability("transfer_overlap", CapabilityStatus.EXPERIMENTAL,
                   "bounded H2D/D2H scheduler with synchronous fallback", "offload"),
        Capability("tp_bulk_overlap_sequence_parallel", CapabilityStatus.EXPERIMENTAL,
                   "TP bulk overlap with explicit SP layout checks", "tp_bulk_overlap"),
        Capability("gradient_bucket_overlap_accumulation", CapabilityStatus.EXPERIMENTAL,
                   "accumulation-boundary gradient bucket scheduling", "gradient_bucket_overlap"),
        Capability("parameter_prefetch_dynamic_control_flow", CapabilityStatus.UNSUPPORTED,
                   "dynamic traces invalidate prefetch and use synchronous fetch", "offload_parameter"),
        Capability("pp_p2p_overlap_gpipe", CapabilityStatus.EXPERIMENTAL,
                   "GPipe warmup/cooldown P2P handle queue", "pp_p2p_overlap"),
        Capability("pp_p2p_overlap_1f1b", CapabilityStatus.EXPERIMENTAL,
                   "non-interleaved 1F1B steady-state P2P handle queue", "pp_p2p_overlap"),
        Capability("transfer_overlap_offload", CapabilityStatus.EXPERIMENTAL,
                   "parameter/activation/optimizer transfer scheduler", "transfer_overlap"),
        Capability("microbatch_interleave", CapabilityStatus.EXPERIMENTAL,
                   "two-microbatch TP interleave; PP/dynamic/offload combinations rejected", "tp_bulk_overlap"),
        Capability("planning_report", CapabilityStatus.STABLE,
                   "deterministic read-only topology/stage recommendations"),
        Capability("planning_apply", CapabilityStatus.EXPERIMENTAL,
                   "explicit reviewed candidate application only", "planning_report"),
        Capability("fp8", CapabilityStatus.UNSUPPORTED,
                   "fp8 scaling, checkpoint and overflow handling are not implemented", "bf16"),
        Capability("compile", CapabilityStatus.UNSUPPORTED,
                   "torch.compile integration is not implemented", "eager"),
    )


def validate_combinations(config: FrameworkConfig, *, world_size: int = 1,
                          torch_module: object | None = None) -> None:
    """Cross-configuration rules only.

    Each config class owns its own invariants (``FrameworkConfig.validate``);
    this owns the rules that span siblings -- "dp_size>1 requires FSDP",
    "microbatch interleave rejects PP", and so on.  Kept separate so an entry
    point that has already validated the config does not validate it twice;
    :func:`validate_capabilities` is the version that does both.
    """
    # DP is a real training dimension only when FSDP owns that group.  TP and
    # PP may be combined with one another and with FSDP because their groups
    # are constructed orthogonally by DeviceMeshManager.
    if config.parallel.dp_size > 1 and not config.fsdp.enabled:
        raise UnsupportedCombinationError(
            "dp_size>1 requires fsdp.enabled=True so gradients are reduced over the DP group"
        )
    if config.compile.enabled:
        raise UnsupportedCombinationError("compile.enabled=True is not implemented")
    overlap = config.overlap
    if overlap.enable_gradient_bucket_overlap and config.fsdp.enabled:
        raise UnsupportedCombinationError(
            "overlap.enable_gradient_bucket_overlap cannot be combined with FSDP; use FSDP's reducer"
        )
    if overlap.enable_optimizer_param_gather_overlap and config.fsdp.enabled:
        raise UnsupportedCombinationError(
            "optimizer parameter gather overlap is unsupported for FSDP without a public prefetch API"
        )
    if overlap.enable_pp_p2p_overlap and config.parallel.pp_size <= 1:
        raise UnsupportedCombinationError("overlap.enable_pp_p2p_overlap requires pp_size>1")
    if overlap.enable_tp_bulk_overlap and config.parallel.tp_size <= 1:
        raise UnsupportedCombinationError("overlap.enable_tp_bulk_overlap requires tp_size>1")
    if overlap.enable_microbatch_interleave:
        if config.parallel.tp_size <= 1:
            raise UnsupportedCombinationError("microbatch interleave requires tp_size>1")
        if config.parallel.pp_size > 1 or config.offload.activation:
            raise UnsupportedCombinationError("microbatch interleave rejects PP and activation offload")
    # CPU bfloat16 is legal for many operators, so it is deliberately NOT rejected
    # globally.  (An earlier version ended with a branch that computed this and
    # then returned without doing anything; removed rather than left as a decoy.)


def validate_capabilities(config: FrameworkConfig, *, world_size: int = 1,
                          torch_module: object | None = None) -> None:
    """The public validation entry point: invariants, then combinations.

    Every entry point that accepts a config should call this one.  It used to be
    that ``Trainer.__init__`` called ``config.validate`` and then called this,
    which called ``config.validate`` again -- so the same config was checked
    twice on every construction, and ``parallelize`` made it three times on the
    ``from_model`` path.
    """
    config.validate(world_size=world_size)
    validate_combinations(config, world_size=world_size, torch_module=torch_module)


def installed_optional_backends() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None
            for name in ("flash_attn", "transformer_engine", "triton", "safetensors")}


def capability_status(name: str) -> Capability:
    for item in capability_matrix():
        if item.name == name:
            return item
    raise KeyError(name)
