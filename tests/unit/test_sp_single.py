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


def test_sequence_parallel_does_not_wrap_dropout():
    """Only norms go inside the sequence-parallel region, never dropout.

    Wrapping dropout there means each rank draws its mask from its own RNG
    stream, and every stream starts at the same place: rank 0 is dropped out with
    the masks of the first block and rank 1 with the SAME masks applied to the
    second block.  Measured against a single-process reference, that is
    ``max|diff| = 7.9e-01``.

    It is also unnecessary.  This style gathers the norm's output back to the
    full sequence, so a dropout that follows a norm sees the whole sequence on
    every rank and reproduces one process exactly (measured 5.96e-08).  Keeping
    it out is what makes it right, and the counterfactual is in
    ``scripts/verify_random_state.py``, which flips this and watches it fail.
    """
    import torch

    from aitrainer.parallel.tp import sequence_parallel_styles

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)
            self.act = torch.nn.Dropout(p=0.5)
            self.inner = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Dropout(p=0.5))

    assert sorted(sequence_parallel_styles(Block())) == ["inner.0", "norm"], (
        "dropout must stay outside the sequence-parallel region")
