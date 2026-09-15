import pytest

from aitrainer import ConfigPreset, ConfigurationError, FrameworkConfig, dry_run


def test_default_config_is_valid_and_serializable():
    config = ConfigPreset.single_gpu()
    config.validate()
    assert FrameworkConfig.from_dict(config.to_dict()) == config


def test_config_is_immutable():
    config = FrameworkConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.seed = 1


@pytest.mark.parametrize("kwargs", [
    {"grad_accumulation_steps": 0},
    {"parallel": {"pp_size": 2}},
    {"compile": {"enabled": True}},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ConfigurationError):
        FrameworkConfig.from_dict(kwargs)


def test_dry_run_reports_capability_boundary():
    result = dry_run()
    assert result["status"] == "ok"
    assert any(item["name"] == "fsdp_full_shard" and item["status"] == "stable" for item in result["capabilities"])


def test_fsdp_preset_enables_only_data_parallel_path():
    config = ConfigPreset.fsdp()
    assert config.fsdp.enabled
    config.replace(parallel=config.parallel).validate()


def test_nested_configs_are_built_generically():
    """Nesting is resolved from the annotations, not from hand-written branches.

    ``from_dict`` used to carry one special case for ``fsdp.mixed_precision``,
    so a second nested config would have needed a second branch and forgetting
    one produced a bare TypeError from the dataclass constructor.
    """
    config = FrameworkConfig.from_dict(
        {"fsdp": {"enabled": True, "mixed_precision": {"enabled": True, "dtype": "float16"}}})
    assert config.fsdp.mixed_precision.enabled is True
    assert config.fsdp.mixed_precision.dtype == "float16"

    # A typo two levels down must still surface as a ConfigurationError, not as
    # the dataclass's own TypeError.
    with pytest.raises(ConfigurationError):
        FrameworkConfig.from_dict({"fsdp": {"mixed_precision": {"enabledd": True}}})


def test_trainer_construction_validates_the_config_once():
    """Constructing a Trainer must validate the config exactly once.

    ``Trainer.__init__`` called ``config.validate`` and then called
    ``validate_capabilities``, which calls it again -- so every construction
    checked the same config twice, and ``from_model`` (which also routes through
    ``parallelize``) made it three times.  Counting the calls is the only way to
    notice the redundancy creeping back.
    """
    torch = pytest.importorskip("torch")
    from aitrainer import FrameworkConfig, Trainer

    calls = []
    original = FrameworkConfig.validate

    def counting_validate(self, world_size=1):
        calls.append(world_size)
        return original(self, world_size=world_size)

    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    FrameworkConfig.validate = counting_validate
    try:
        Trainer(model, optimizer, config=FrameworkConfig(device="cpu", seed=5))
    finally:
        FrameworkConfig.validate = original
    assert len(calls) == 1, f"config validated {len(calls)} times during construction"


def test_fsdp_refuses_forward_prefetch_without_a_trace():
    """The second layer must agree with the first, not silently disagree.

    ``config.validate`` rejects ``forward_prefetch`` without a finalised
    execution trace.  ``FSDPWrapper`` ANDed the flag away instead, so a caller
    that bypassed config validation was told nothing -- "prefetch was refused"
    and "prefetch is off" are different things to report.
    """
    torch = pytest.importorskip("torch")
    from aitrainer import FSDPConfig
    from aitrainer.parallel.fsdp import FSDPConfigurationError, FSDPWrapper

    with pytest.raises(FSDPConfigurationError):
        FSDPWrapper(torch.nn.Linear(3, 2),
                    config=FSDPConfig(forward_prefetch=True, execution_trace_complete=False))
