import pytest

torch = pytest.importorskip("torch")

from aitrainer import SequenceParallelLayerNorm, distributed_attention


def test_sequence_parallel_tp1_preserves_shape_and_values():
    torch.manual_seed(3)
    value = torch.randn(2, 5, 8)
    module = SequenceParallelLayerNorm(8)
    expected = torch.nn.LayerNorm(8)
    expected.load_state_dict(module.norm.state_dict())
    assert torch.allclose(module(value), expected(value))


def test_ulysses_attention_tp1_matches_eager_attention_shape():
    q = torch.randn(2, 4, 2, 4, requires_grad=True)
    output = distributed_attention(q, q, q)
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None
