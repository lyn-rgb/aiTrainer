import pytest

from aitrainer.data import BatchContract, DataContractError, StatefulSampler


def test_batch_contract_validates_token_normalization():
    contract = BatchContract(microbatch_count=2, token_normalization="token_mean")
    with pytest.raises(DataContractError):
        contract.normalize_loss(type("Scalar", (), {"ndim": 0})())


def test_sampler_state_round_trip():
    sampler = StatefulSampler(object(), seed=9)
    sampler.set_epoch(3)
    state = sampler.state_dict()
    other = StatefulSampler(object())
    other.load_state_dict(state)
    assert other.epoch == 3 and other.seed == 9
