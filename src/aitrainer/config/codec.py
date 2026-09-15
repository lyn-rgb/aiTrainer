"""Turning plain mappings into config objects.

Kept generic on purpose.  ``from_dict`` used to hand-code one special case --
``if name == "fsdp" and "mixed_precision" in raw`` -- so adding any second
nested config would have needed a second branch, and forgetting one produced a
``TypeError`` from the dataclass constructor rather than a config error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from .errors import ConfigurationError


def nested_config_types(cls: Any) -> dict[str, Any]:
    """Field name -> config dataclass, for every field that holds another config."""
    try:
        hints = get_type_hints(cls)
    except NameError:                      # pragma: no cover - forward refs
        return {}
    found: dict[str, Any] = {}
    for item in fields(cls):
        hint = hints.get(item.name)
        origin = get_origin(hint)
        candidates = get_args(hint) if origin is not None else (hint,)
        for candidate in candidates:
            if isinstance(candidate, type) and is_dataclass(candidate) and candidate is not cls:
                found[item.name] = candidate
                break
    return found


def build_config(cls: Any, raw: Mapping[str, Any], name: str = "") -> Any:
    """Instantiate ``cls`` from a mapping, recursing into nested configs."""
    nested = nested_config_types(cls)
    kwargs = dict(raw)
    for key, sub_type in nested.items():
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            kwargs[key] = build_config(sub_type, value, f"{name}.{key}" if name else key)
    return cls(**kwargs)


def instantiate(cls: Any, raw: Any, name: str) -> Any:
    """``build_config`` with the errors a caller of ``from_dict`` can act on."""
    if raw is None:
        # `{"parallel": null}` used to crash later with a bare AttributeError on
        # None.dp_size.
        raise ConfigurationError(f"{name} must be an object, not null")
    if is_dataclass(raw):
        return raw
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{name} must be an object")
    try:
        return build_config(cls, raw, name)
    except ConfigurationError:
        raise
    except TypeError as exc:
        # A typo'd nested key surfaced as `ParallelConfig.__init__() got an
        # unexpected keyword argument 'tp_sizee'`, leaking the class name and
        # bypassing `except ConfigurationError` handlers.
        raise ConfigurationError(f"{name} has an unknown or invalid field: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"invalid {name} configuration: {exc}") from exc
