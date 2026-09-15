from aitrainer.config import ConfigurationError, OffloadConfig


def test_offload_disabled_is_valid():
    OffloadConfig().validate()


def test_offload_requires_mode():
    try:
        OffloadConfig(enabled=True).validate()
    except ConfigurationError:
        return
    raise AssertionError("enabled offload without a mode must fail")


def test_parameter_activation_are_explicitly_rejected():
    try:
        OffloadConfig(enabled=True, parameter=True, activation=True).validate()
    except ConfigurationError:
        return
    raise AssertionError("unvalidated parameter+activation combination must fail")
