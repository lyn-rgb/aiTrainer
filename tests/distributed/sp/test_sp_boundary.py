"""SP integration entry points for fixed-shape Gloo/NCCL runs."""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, validate


def test_sp_requires_tp():
    with pytest.raises(ValueError):
        FrameworkConfig.from_dict({"parallel": {"sp_backend": "megatron"}})


def test_tp_sp_configuration_is_explicit():
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": 2, "sp_backend": "megatron"}})
    validate(config, world_size=2)


def test_ulysses_sp_backend_is_refused_rather_than_silently_reinterpreted():
    """The value has to fail loudly, because accepting it did something else.

    ``parallelize`` installs the norm sequence sharding for ANY non-'none' value
    -- there is one SP implementation -- so ``sp_backend="ulysses"`` ran
    Megatron-style SP while its name, its comments and the validation matrix all
    said Ulysses.  Measured: the two entries in ``scripts/parallel_matrix.py``
    that differed only in this field produced bit-identical losses, and one of
    them was described as "the Ulysses/TP path".

    The value stays in the enum, like ``compile.enabled``, so the refusal can
    explain itself and name the way out.  Deleting it would leave a user with
    "unknown value 'ulysses'" and no indication of what to use instead.
    """
    from aitrainer import ParallelConfig

    # Both entry points: from_dict validates as it parses, so a user reaches the
    # refusal there; validate() is what a hand-built config goes through.
    with pytest.raises(ValueError) as from_dict_failure:
        FrameworkConfig.from_dict({"parallel": {"tp_size": 2, "sp_backend": "ulysses"}})
    with pytest.raises(ValueError) as validate_failure:
        validate(FrameworkConfig(parallel=ParallelConfig(tp_size=2, sp_backend="ulysses")),
                 world_size=2)

    for caught in (from_dict_failure, validate_failure):
        message = str(caught.value)
        assert "not wired into the model path" in message
        assert "'megatron'" in message, "the error must say what to use instead"
        assert "UlyssesAttention" in message, (
            "the error must name the explicit API that does exist")

    # The value it points at still works, at the same world size.
    validate(FrameworkConfig.from_dict({"parallel": {"tp_size": 2, "sp_backend": "megatron"}}),
             world_size=2)
