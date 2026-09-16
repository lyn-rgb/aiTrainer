"""Immutable configuration objects and startup validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

from ..core.dtypes import dtype_name
from .codec import instantiate
from .errors import ConfigurationError


@dataclass(frozen=True)
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sp_backend: Literal["none", "ulysses", "megatron"] = "none"
    pp_schedule: Literal["none", "gpipe", "1f1b"] = "none"
    num_microbatches: int = 1

    def validate(self, world_size: int = 1) -> None:
        for name in ("dp_size", "tp_size", "pp_size", "num_microbatches"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(f"parallel.{name}={value!r} must be a positive integer")
        if self.sp_backend not in {"none", "ulysses", "megatron"}:
            raise ConfigurationError(f"unknown parallel.sp_backend={self.sp_backend!r}")
        if self.sp_backend != "none" and self.tp_size <= 1:
            raise ConfigurationError("parallel.sp_backend requires tp_size>1")
        if self.sp_backend == "ulysses":
            # Refused rather than accepted-and-ignored.  `parallelize` installs
            # the norm sequence sharding for any non-'none' value (there is one
            # SP implementation), so this setting used to run Megatron-style SP
            # while its name, its comments and the validation matrix all said
            # Ulysses -- measured: `fsdp_sp` and `fsdp_sp_tp` in
            # scripts/parallel_matrix.py, which differed only in this field,
            # produced bit-identical losses.
            #
            # Wiring it up is not a rename: the head exchange needs all_to_all,
            # Gloo has no all_to_all, so every world_size>1 Ulysses path is
            # CUDA/NCCL-only and cannot be verified on this host.  Keeping the
            # value in the enum and refusing it is the same treatment
            # `compile.enabled` gets, and it leaves the wire-in point visible.
            raise ConfigurationError(
                "parallel.sp_backend='ulysses' is not wired into the model path: the "
                "head exchange needs NCCL (Gloo has no all_to_all), so parallelize() "
                "installs the norm sequence sharding for any non-'none' value. Use "
                "'megatron', or call UlyssesAttention explicitly on the projections "
                "you want exchanged.")
        if self.pp_schedule not in {"none", "gpipe", "1f1b"}:
            raise ConfigurationError(f"unknown parallel.pp_schedule={self.pp_schedule!r}")
        if self.pp_size == 1 and self.pp_schedule != "none":
            raise ConfigurationError("parallel.pp_schedule requires pp_size>1")
        if self.pp_size > 1 and self.pp_schedule == "none":
            raise ConfigurationError("parallel.pp_schedule must be 'gpipe' or '1f1b' when pp_size>1")
        expected = self.dp_size * self.tp_size * self.pp_size
        if expected != world_size:
            raise ConfigurationError(
                f"parallel dimensions product={expected} does not match world_size={world_size}"
            )
        if self.pp_schedule == "1f1b" and self.num_microbatches < self.pp_size:
            raise ConfigurationError("1f1b requires num_microbatches >= pp_size")


@dataclass(frozen=True)
class PrecisionConfig:
    """Per-tensor-class dtypes, each one consumed by the trainer.

    Parameter *storage* dtype is controlled by ``FSDPConfig.mixed_precision.dtype``,
    and so is the reduction dtype FSDP uses: FSDP2 sets both from that one field.
    There is deliberately neither a ``param_dtype`` nor a ``reduce_dtype`` here.
    Each used to exist with no consumer at all and silently did nothing when set,
    and each was deleted for that reason.  ``reduce_dtype`` was the subtler of the
    two: it genuinely reached ``RowParallelLinear``'s reduction before the
    DTensor migration, so it looked wired for several rounds after that
    migration had dropped the connection on the floor -- the field, its
    validation, two presets that set it and a docstring claiming the wiring all
    survived the code that did the work.
    """

    compute_dtype: Any = "float32"
    grad_dtype: Any = "float32"
    optimizer_dtype: Any = "float32"
    use_grad_scaler: bool | None = None

    def validate(self) -> None:
        names = ("compute_dtype", "grad_dtype", "optimizer_dtype")
        allowed = {"float32", "float16", "bfloat16"}
        for name in names:
            value = getattr(self, name)
            if isinstance(value, str) and value not in allowed:
                raise ConfigurationError(f"precision.{name}={value!r} is not supported")
        if self.use_grad_scaler is not None and not isinstance(self.use_grad_scaler, bool):
            raise ConfigurationError("precision.use_grad_scaler must be bool or None")


@dataclass(frozen=True)
class CompileConfig:
    """``torch.compile`` is not implemented; ``enabled`` exists only to reject it.

    ``mode``/``fullgraph``/``dynamic`` used to sit here, validated and documented
    but read by nothing -- they are gone rather than left implying the feature is
    configurable.
    """

    enabled: bool = False

    def validate(self) -> None:
        if self.enabled:
            raise ConfigurationError(
                "compile.enabled is not implemented; keep it disabled (this is a "
                "documented gap, not a version limit)")


@dataclass(frozen=True)
class MixedPrecisionConfig:
    enabled: bool = False
    dtype: Any = "bfloat16"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("fsdp.mixed_precision.enabled must be bool")
        dtype = self.dtype
        if isinstance(dtype, str):
            if dtype_name(dtype) not in {"float16", "bfloat16", "float32"}:
                # Previously unvalidated: a garbage dtype reached FSDP and failed
                # deep inside torch (or silently mis-cast).
                raise ConfigurationError(f"fsdp.mixed_precision.dtype={dtype!r} is not supported")
        elif "torch." not in str(dtype):
            # A real torch.dtype reprs as "torch.float16"; anything else is a
            # mistake better caught here than inside fully_shard.
            raise ConfigurationError(f"fsdp.mixed_precision.dtype={dtype!r} is not a torch dtype")


@dataclass(frozen=True)
class FSDPConfig:
    """The two FSDP2 knobs this framework actually forwards.

    Five fields used to live here.  ``sharding`` was FSDP1's strategy enum
    (FSDP2 shards a single way); ``limit_all_gathers`` and ``forward_prefetch``
    were FSDP1 scheduler flags with no FSDP2 counterpart; and
    ``execution_trace_complete`` existed only to gate ``forward_prefetch``.
    None of them reached anything, so a config could set them and train
    unsharded-and-unprefetched with no signal.  ``cpu_offload`` was the same
    shape but louder -- it raised, and pointed at ``FrameworkConfig.offload``,
    which is the mechanism that does exist.

    ``MixedPrecisionPolicy`` has no ``buffer_dtype``, so it is not reachable
    from here either; see ``parallel.fsdp``.
    """

    enabled: bool = False
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("fsdp.enabled must be bool")
        self.mixed_precision.validate()


@dataclass(frozen=True)
class OffloadConfig:
    enabled: bool = False
    optimizer: bool = False
    parameter: bool = False
    activation: bool = False
    pin_memory: bool = True
    max_cpu_bytes: int = 2 * 1024**3
    activation_threshold_bytes: int = 1 * 1024**2
    keep_last_activations: int = 1

    def validate(self) -> None:
        for name in ("enabled", "optimizer", "parameter", "activation", "pin_memory"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"offload.{name} must be bool")
        for name in ("max_cpu_bytes", "activation_threshold_bytes", "keep_last_activations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigurationError(f"offload.{name} must be an integer")
        if self.max_cpu_bytes < 1 or self.activation_threshold_bytes < 0:
            raise ConfigurationError("offload memory limits must be non-negative/positive")
        if self.keep_last_activations < 0:
            raise ConfigurationError("offload.keep_last_activations must be non-negative")
        if self.enabled and not (self.optimizer or self.parameter or self.activation):
            raise ConfigurationError("offload.enabled requires optimizer, parameter, or activation mode")
        if not self.enabled and (self.optimizer or self.parameter or self.activation):
            raise ConfigurationError("offload modes require offload.enabled=True; refusing silent disable")
        if self.activation and self.parameter:
            raise ConfigurationError("parameter+activation offload requires an explicit validated combination")


@dataclass(frozen=True)
class AdvancedOverlapConfig:
    """Independent overlap switches and bounded in-flight resource budgets.

    The ``enable_*`` switches are **rejection-only as of this version**: turning
    one on does not turn the corresponding feature on, it only makes startup
    reject combinations that genuinely cannot work (see
    ``capability.validate_capabilities``).  They are kept deliberately -- removing
    them would silently widen the set of accepted configurations, which is a
    behaviour change disguised as cleanup.

    Byte budgets set to ``0`` mean automatic/unbounded for that optional class;
    operation count remains a strictly positive safety limit.
    """
    enable_tp_bulk_overlap: bool = False
    enable_gradient_bucket_overlap: bool = False
    enable_optimizer_param_gather_overlap: bool = False
    enable_pp_p2p_overlap: bool = False
    enable_transfer_overlap: bool = False
    enable_microbatch_interleave: bool = False
    parameter_prefetch_bytes: int = 0
    max_inflight_ops: int = 2
    max_inflight_bytes: int = 0
    max_transfer_bytes: int = 0
    drain_timeout_s: float = 60.0

    def validate(self) -> None:
        for name in ("enable_tp_bulk_overlap", "enable_gradient_bucket_overlap",
                     "enable_optimizer_param_gather_overlap", "enable_pp_p2p_overlap", "enable_transfer_overlap",
                     "enable_microbatch_interleave"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"overlap.{name} must be bool")
        for name in ("parameter_prefetch_bytes", "max_inflight_ops", "max_inflight_bytes",
                     "max_transfer_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigurationError(f"overlap.{name} must be a non-negative integer")
        if self.max_inflight_ops < 1:
            raise ConfigurationError("overlap.max_inflight_ops must be positive")
        if self.drain_timeout_s < 0:
            raise ConfigurationError("overlap.drain_timeout_s must be non-negative")


@dataclass(frozen=True)
class PlanningConfig:
    """Opt-in automatic planning; reports never rewrite explicit topology silently."""

    enabled: bool = False
    allow_rewrite: bool = False
    stage_policy: Literal["uniform_layers", "parameter_count", "flops", "memory_balanced"] = "uniform_layers"
    max_candidates: int = 8
    seed: int = 42

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.allow_rewrite, bool):
            raise ConfigurationError("planning.enabled and planning.allow_rewrite must be bool")
        if self.stage_policy not in {"uniform_layers", "parameter_count", "flops", "memory_balanced"}:
            raise ConfigurationError(f"unknown planning.stage_policy={self.stage_policy!r}")
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or self.max_candidates < 1:
            raise ConfigurationError("planning.max_candidates must be a positive integer")


@dataclass(frozen=True)
class FrameworkConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    overlap: AdvancedOverlapConfig = field(default_factory=AdvancedOverlapConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    grad_accumulation_steps: int = 1
    grad_clip_norm: float | None = None
    seed: int = 42
    log_interval: int = 10
    device: str = "auto"

    def validate(self, world_size: int = 1) -> None:
        if self.grad_accumulation_steps < 1:
            raise ConfigurationError("grad_accumulation_steps must be >= 1")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ConfigurationError("grad_clip_norm must be > 0 or None")
        if self.log_interval < 1:
            raise ConfigurationError("log_interval must be >= 1")
        self.parallel.validate(world_size)
        self.precision.validate()
        self.compile.validate()
        self.fsdp.validate()
        self.offload.validate()
        self.overlap.validate()
        self.planning.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> FrameworkConfig:
        """Return a new config; nested fields can be replaced explicitly."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameworkConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown configuration fields: {sorted(unknown)}")
        values = dict(data)
        for name, typ in (("parallel", ParallelConfig), ("precision", PrecisionConfig),
                          ("compile", CompileConfig), ("fsdp", FSDPConfig),
                          ("offload", OffloadConfig), ("overlap", AdvancedOverlapConfig), ("planning", PlanningConfig)):
            raw = values.get(name)
            if raw is None:
                if name in values:
                    raise ConfigurationError(f"{name} must be an object, not null")
                continue
            values[name] = instantiate(typ, raw, name)
        result = cls(**values)
        # Configuration parsing happens before Runtime knows the launched
        # world size. Validate structural constraints against the declared
        # mesh product; Runtime performs the final product/world check.
        result.validate(world_size=result.parallel.dp_size * result.parallel.tp_size * result.parallel.pp_size)
        return result
