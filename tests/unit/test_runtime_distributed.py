import pytest

from aitrainer.runtime import DistributedInitializationError, Runtime


def test_runtime_reads_torchrun_environment(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Avoid rendezvous in the unit test while still exercising rank parsing.
    runtime = Runtime(device="cpu", init_process_group=False)
    assert runtime.rank == 1 and runtime.world_size == 2
    runtime.close()


def test_runtime_rejects_invalid_rank_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(DistributedInitializationError):
        Runtime(device="cpu", init_process_group=False)
