"""FSDP and orthogonal TP/SP/PP integration entry points.

Run the real test with ``torchrun --standalone --nproc_per_node=2 -m pytest
tests/distributed/fsdp`` in a CUDA/NCCL or CPU/Gloo environment.  The local
environment may legitimately report the test as unavailable when torch is not
installed; it must not silently change the requested configuration.
"""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import ConfigPreset, validate


def test_fsdp_configuration_accepts_dp_only(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = ConfigPreset.fsdp()
    config = config.replace(parallel=type(config.parallel)(dp_size=2))
    validate(config, world_size=2)


def test_fsdp_tp_combination_is_accepted_at_startup():
    config = ConfigPreset.fsdp()
    validate(config.replace(parallel=type(config.parallel)(dp_size=1, tp_size=2)), world_size=2)


@pytest.mark.parametrize(
    ("dp_size", "tp_size", "pp_size", "sp_backend", "pp_schedule", "microbatches", "world_size"),
    [
        (2, 2, 1, "megatron", "none", 1, 4),
        (2, 2, 2, "megatron", "1f1b", 2, 8),
    ],
)
def test_fsdp_sp_tp_pp_combinations_are_accepted_at_startup(
    dp_size, tp_size, pp_size, sp_backend, pp_schedule, microbatches, world_size
):
    parallel = type(ConfigPreset.fsdp().parallel)(
        dp_size=dp_size, tp_size=tp_size, pp_size=pp_size,
        sp_backend=sp_backend, pp_schedule=pp_schedule,
        num_microbatches=microbatches,
    )
    validate(ConfigPreset.fsdp().replace(parallel=parallel), world_size=world_size)
