"""The configuration error type, kept apart so codec and schema can share it."""

from __future__ import annotations


class ConfigurationError(ValueError):
    """Raised when a configuration cannot be safely executed."""
