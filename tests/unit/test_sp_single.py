"""Single-process contracts for the sequence-parallel and Ulysses paths.

Sequence parallelism itself can only be checked where there is more than one
rank to shard across -- see ``tests/unit/test_multirank_gloo.py``.  What is
checkable here is the plan: which modules get a style, and which layout the
projections are told to use.
"""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import distributed_attention


def test_ulysses_attention_tp1_matches_eager_attention_shape():
    q = torch.randn(2, 4, 2, 4, requires_grad=True)
    output = distributed_attention(q, q, q)
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None


def test_ulysses_attention_is_dense_at_one_rank():
    """At world=1 the head exchange is the identity, so it must equal SDPA exactly.

    The reference is not ``SDPA(q, q, q)``.  Inputs are ``[B, L, H, D]`` and
    ``scaled_dot_product_attention`` attends over its *last two* dimensions, so
    calling it directly attends across **heads** -- measured as a max error of
    1.24 against the same inputs.  ``distributed_attention`` transposes heads
    and sequence so the attention runs over the sequence axis, which is what the
    name means; written out, that reference matches to exactly 0.0.  A test that
    reached for the obvious baseline would have asserted the wrong convention.
    """
    from torch.nn import functional

    torch.manual_seed(5)
    q = torch.randn(2, 3, 4, 8)
    output = distributed_attention(q, q, q)
    heads_first = q.transpose(1, 2)
    expected = functional.scaled_dot_product_attention(
        heads_first, heads_first, heads_first).transpose(1, 2)
    assert torch.equal(output, expected)
