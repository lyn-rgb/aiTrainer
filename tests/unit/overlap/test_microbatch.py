import pytest

from aitrainer.overlap import MicrobatchInterleaveScheduler


def test_microbatch_interleave_runs_both_directions_and_normalizes_loss():
    backward = []
    scheduler = MicrobatchInterleaveScheduler(tp_size=2)
    result = scheduler.run([1, 2], forward_fn=lambda value: value + 1,
                           backward_fn=lambda activation, loss: backward.append((activation, loss)))
    assert result.microbatches == 2 and result.normalized_loss == 2.5
    assert backward == [(2, 2), (3, 3)]


def test_microbatch_interleave_rejects_unsupported_combinations():
    with pytest.raises(ValueError): MicrobatchInterleaveScheduler(tp_size=2, pp_size=2)
    with pytest.raises(ValueError): MicrobatchInterleaveScheduler(tp_size=2, activation_offload=True)
    with pytest.raises(ValueError): MicrobatchInterleaveScheduler(tp_size=1)
