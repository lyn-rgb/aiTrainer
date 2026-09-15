"""SP integration entry points for fixed-shape Gloo/NCCL runs."""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, validate


def test_sp_requires_tp():
    with pytest.raises(ValueError):
        FrameworkConfig.from_dict({"parallel": {"sp_backend": "ulysses"}})


def test_tp_sp_configuration_is_explicit():
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": 2, "sp_backend": "ulysses"}})
    validate(config, world_size=2)
