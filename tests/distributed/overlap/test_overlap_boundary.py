import pytest

from aitrainer.capability import UnsupportedCombinationError, validate_capabilities
from aitrainer.config import AdvancedOverlapConfig, FrameworkConfig, ParallelConfig


def test_overlap_switches_require_matching_parallel_axis():
    config = FrameworkConfig(overlap=AdvancedOverlapConfig(enable_tp_bulk_overlap=True))
    with pytest.raises(UnsupportedCombinationError):
        validate_capabilities(config)
    config = FrameworkConfig(parallel=ParallelConfig(tp_size=2, dp_size=1),
                             overlap=AdvancedOverlapConfig(enable_tp_bulk_overlap=True))
    validate_capabilities(config, world_size=2)
