"""Immutable configuration objects and startup validation.

A package rather than a module so that the schema, the mapping codec and the
error type can be read separately.  Every public name is re-exported here, so
``from aitrainer.config import FrameworkConfig`` is unchanged.
"""

from __future__ import annotations

from .codec import build_config, instantiate, nested_config_types
from .errors import ConfigurationError
from .schema import (
    AdvancedOverlapConfig,
    CompileConfig,
    FrameworkConfig,
    FSDPConfig,
    MixedPrecisionConfig,
    OffloadConfig,
    ParallelConfig,
    PlanningConfig,
    PrecisionConfig,
)

__all__ = [
    "AdvancedOverlapConfig",
    "CompileConfig",
    "ConfigurationError",
    "FSDPConfig",
    "FrameworkConfig",
    "MixedPrecisionConfig",
    "OffloadConfig",
    "ParallelConfig",
    "PlanningConfig",
    "PrecisionConfig",
    "build_config",
    "instantiate",
    "nested_config_types",
]
