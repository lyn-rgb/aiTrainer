"""Explicit, reviewable configuration presets."""

from .config import FSDPConfig, FrameworkConfig, ParallelConfig, PrecisionConfig


class ConfigPreset:
    @staticmethod
    def single_gpu() -> FrameworkConfig:
        return FrameworkConfig()

    @staticmethod
    def fsdp() -> FrameworkConfig:
        return FrameworkConfig(parallel=ParallelConfig(dp_size=1), fsdp=FSDPConfig(enabled=True))

    @staticmethod
    def tp(*, tp_size: int = 2) -> FrameworkConfig:
        return FrameworkConfig(parallel=ParallelConfig(tp_size=tp_size))

    @staticmethod
    def tp_fsdp(*, tp_size: int = 2) -> FrameworkConfig:
        return FrameworkConfig(parallel=ParallelConfig(tp_size=tp_size), fsdp=FSDPConfig(enabled=True))

    @staticmethod
    def pp(*, pp_size: int = 2, schedule: str = "gpipe", num_microbatches: int = 2) -> FrameworkConfig:
        return FrameworkConfig(parallel=ParallelConfig(pp_size=pp_size, pp_schedule=schedule,
                                                        num_microbatches=num_microbatches))

    @staticmethod
    def memory_saving() -> FrameworkConfig:
        """Favour memory: bf16 compute with bf16 OPTIMIZER STATE, fp32 gradients.

        Adam holds two moments per parameter, so bf16 optimizer state halves the
        dominant non-parameter footprint -- the deliberate trade for this preset
        (``cast_optimizer_state`` preserves the integer step counter).

        ``grad_dtype`` stays float32: the gradient/parameter dtype check
        (``precision.cast_gradients``) requires them to match, and ``param_dtype``
        is INERT -- parameter storage is controlled by
        ``FSDPConfig.mixed_precision.dtype`` -- so a bf16 ``grad_dtype`` is
        impossible to satisfy and made this preset raise on the first step.
        """
        return FrameworkConfig(precision=PrecisionConfig(
            param_dtype="bfloat16", compute_dtype="bfloat16", grad_dtype="float32",
            reduce_dtype="bfloat16", optimizer_dtype="bfloat16"))

    @staticmethod
    def performance() -> FrameworkConfig:
        """Favour speed and stability: bf16 compute, fp32 gradients and optimizer.

        Previously this and `memory_saving` returned byte-identical configs, so
        comparing the two presets measured nothing.  The live difference is
        ``optimizer_dtype`` (fp32 here, bf16 in ``memory_saving``).
        """
        return FrameworkConfig(precision=PrecisionConfig(
            param_dtype="bfloat16", compute_dtype="bfloat16", grad_dtype="float32",
            reduce_dtype="bfloat16", optimizer_dtype="float32"))
