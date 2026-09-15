"""Immutable configuration objects and startup validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, Literal

from .core.dtypes import dtype_name


class ConfigurationError(ValueError):
    """Raised when a configuration cannot be safely executed."""


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
    """Per-tensor-class dtypes.

    ``compute_dtype``, ``grad_dtype`` and ``optimizer_dtype`` are consumed by the
    trainer.  ``reduce_dtype`` is threaded into the TP row projections.

    Parameter *storage* dtype is controlled by ``FSDPConfig.mixed_precision.dtype``;
    there is deliberately no ``param_dtype`` here, because one used to exist with no
    consumer at all and silently did nothing when set.
    """

    compute_dtype: Any = "float32"
    grad_dtype: Any = "float32"
    reduce_dtype: Any = "float32"
    optimizer_dtype: Any = "float32"
    use_grad_scaler: bool | None = None

    def validate(self) -> None:
        names = ("compute_dtype", "grad_dtype", "reduce_dtype", "optimizer_dtype")
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
            # mistake better caught here than inside FullyShardedDataParallel.
            raise ConfigurationError(f"fsdp.mixed_precision.dtype={dtype!r} is not a torch dtype")


@dataclass(frozen=True)
class FSDPConfig:
    enabled: bool = False
    sharding: Literal["FULL_SHARD"] = "FULL_SHARD"
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    execution_trace_complete: bool = False
    cpu_offload: bool = False

    def validate(self) -> None:
        self.mixed_precision.validate()
        if self.sharding != "FULL_SHARD":
            raise ConfigurationError("fsdp.sharding must be 'FULL_SHARD'")
        if self.cpu_offload:
            raise ConfigurationError(
                "fsdp.cpu_offload is not wired to the Batch 6 OffloadManager; "
                "use FrameworkConfig.offload explicitly"
            )
        if self.forward_prefetch and not self.execution_trace_complete:
            raise ConfigurationError(
                "fsdp.forward_prefetch requires execution_trace_complete=True; "
                "dynamic execution must not enable prefetch"
            )


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
    enable_parameter_prefetch: bool = False
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
        for name in ("enable_tp_bulk_overlap", "enable_gradient_bucket_overlap", "enable_parameter_prefetch",
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
                    # `{"parallel": null}` used to crash later with a bare
                    # AttributeError on None.dp_size.
                    raise ConfigurationError(f"{name} must be an object, not null")
                continue
            if is_dataclass(raw):
                continue
            if not isinstance(raw, Mapping):
                raise ConfigurationError(f"{name} must be an object")
            try:
                if name == "fsdp" and isinstance(raw.get("mixed_precision"), Mapping):
                    raw = dict(raw)
                    raw["mixed_precision"] = MixedPrecisionConfig(**raw["mixed_precision"])
                values[name] = typ(**raw)
            except ConfigurationError:
                raise
            except TypeError as exc:
                # A typo'd nested key surfaced as `ParallelConfig.__init__() got
                # an unexpected keyword argument 'tp_sizee'`, leaking the class
                # name and bypassing `except ConfigurationError` handlers.
                raise ConfigurationError(f"{name} has an unknown or invalid field: {exc}") from exc
            except ValueError as exc:
                raise ConfigurationError(f"invalid {name} configuration: {exc}") from exc
        result = cls(**values)
        # Configuration parsing happens before Runtime knows the launched
        # world size. Validate structural constraints against the declared
        # mesh product; Runtime performs the final product/world check.
        result.validate(world_size=result.parallel.dp_size * result.parallel.tp_size * result.parallel.pp_size)
        return result
