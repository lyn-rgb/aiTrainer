"""Unit contracts for orthogonal mesh composition and pipeline peer mapping."""

from aitrainer import FSDPConfig, FrameworkConfig, ParallelConfig, validate
from aitrainer.parallel.pp_p2p import P2PCommunicator


def test_fsdp_sp_tp_pp_configuration_is_validated_as_supported():
    parallel = ParallelConfig(dp_size=2, tp_size=2, pp_size=2,
                              sp_backend="ulysses", pp_schedule="gpipe",
                              num_microbatches=2)
    validate(FrameworkConfig(parallel=parallel, fsdp=FSDPConfig(enabled=True)), world_size=8)


def test_pipeline_peer_mapping_uses_global_ranks_from_the_pp_group():
    communicator = P2PCommunicator(pp_rank=0, pp_size=2, pp_ranks=(1, 5))
    assert communicator._peer(True) == 5
    communicator = P2PCommunicator(pp_rank=1, pp_size=2, pp_ranks=(1, 5))
    assert communicator._peer(False) == 1
