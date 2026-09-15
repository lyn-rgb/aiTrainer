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
