"""Tensor sizing and container traversal.

``numel() * element_size()`` was written out at roughly fifteen sites, and two
recursive walkers existed with *different* predicates:

* ``trainer._move`` moves anything exposing ``.to`` -- including a whole Module --
  and recurses into mappings, tuples and lists.
* ``offload.optimizer._map_tensors`` maps only floating-point tensors, and checks
  the container cases first.

Merging them naively would change behaviour, so :func:`walk` takes the predicate
as an argument and the two callers keep their own semantics on top of it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def tensor_bytes(value: Any) -> int:
    """Bytes held by a tensor, or ``0`` for anything without a real size.

    ``element_size`` is a method, not an attribute, so the naive
    ``getattr(t, "element_size", 4)`` idiom that this replaces would have handed
    back a bound method instead of a number for tensor-likes that define it.
    """
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return 0
    return int(numel()) * int(element_size())


def walk(value: Any, predicate: Callable[[Any], bool], fn: Callable[[Any], Any]) -> Any:
    """Rebuild ``value``, applying ``fn`` to every leaf ``predicate`` accepts.

    Containers are rebuilt in kind (dict stays dict, tuple stays tuple) so the
    batch contract downstream is unchanged.
    """
    if isinstance(value, Mapping):
        return {key: walk(item, predicate, fn) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(walk(item, predicate, fn) for item in value)
    if isinstance(value, list):
        return [walk(item, predicate, fn) for item in value]
    return fn(value) if predicate(value) else value


def _movable(value: Any) -> bool:
    return hasattr(value, "to")


def move_to(value: Any, device: Any) -> Any:
    """Move every movable leaf to ``device``, recursing through containers.

    The predicate is ``hasattr(value, "to")`` -- a tensor, a Module or anything
    else device-movable -- which is what ``trainer._move`` did.
    """
    return walk(value, _movable, lambda item: item.to(device))


def _tensor_like(value: Any) -> bool:
    """Tensor-likes, by attribute presence -- NOT by dtype.

    The name this replaces was ``_map_tensors`` and its guard read
    ``hasattr(value, "is_floating_point")``, which tests that the attribute
    *exists*, not that it is true.  So integer tensors are mapped too, and that
    is load-bearing: the caller moves any tensor whose device differs, and Adam's
    ``step`` counter is an int64 tensor whose device must follow the rest of the
    state.  Narrowing this to ``is_floating_point() is True`` would strand that
    counter on the wrong device, so the permissive form is kept deliberately.
    """
    return hasattr(value, "is_floating_point") and hasattr(value, "to")


def map_tensors(value: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every tensor-like leaf, recursing through containers."""
    return walk(value, _tensor_like, fn)
