"""Boundary tests that do not pretend to validate CUDA transfer performance."""

import pytest

from aitrainer import FrameworkConfig, OffloadConfig


def test_parameter_activation_combination_is_not_silent():
    with pytest.raises(ValueError):
        FrameworkConfig(offload=OffloadConfig(enabled=True, parameter=True, activation=True)).validate()


def test_offload_requires_explicit_enable():
    with pytest.raises(ValueError):
        FrameworkConfig(offload=OffloadConfig(optimizer=True)).validate()
