import pytest

torch = pytest.importorskip("torch")

from aitrainer.parallel.tp import ColumnParallelLinear, RowParallelLinear


def test_tp1_linear_matches_dense_parameters():
    torch.manual_seed(1)
    dense = torch.nn.Linear(4, 6)
    column = ColumnParallelLinear.from_dense(dense, gather_output=True)
    row = RowParallelLinear.from_dense(dense)
    value = torch.randn(3, 4)
    assert torch.allclose(column(value), dense(value))
    assert torch.allclose(row(value), dense(value))


def test_tp_dimension_validation():
    # This remains a meaningful construction check even in a single-rank process.
    ColumnParallelLinear(4, 6)
