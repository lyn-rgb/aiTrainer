"""TP integration tests for a torchrun CPU/Gloo or CUDA/NCCL environment."""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, validate
from aitrainer.parallel.tp import ColumnParallelLinear, RowParallelLinear


def test_tp_configuration_is_dp_pp_exclusive():
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": 2}})
    validate(config, world_size=2)


def test_tp_modules_construct_at_tp1():
    column = ColumnParallelLinear(4, 8)
    row = RowParallelLinear(8, 4)
    value = torch.randn(2, 4)
    assert column(value).shape == (2, 8)
    assert row(column(value)).shape == (2, 4)
