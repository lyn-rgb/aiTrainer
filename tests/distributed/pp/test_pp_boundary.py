"""PP integration boundary tests for torchrun Gloo/NCCL environments."""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, validate


def test_pp_configuration_is_pure_pp_in_batch4():
    config = FrameworkConfig.from_dict({
        "parallel": {"pp_size": 2, "pp_schedule": "gpipe", "num_microbatches": 2}
    })
    validate(config, world_size=2)
