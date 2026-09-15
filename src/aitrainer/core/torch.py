"""The single policy for optional PyTorch use.

``import aitrainer`` has always worked with torch absent, but it worked by
accident: nine different shapes of ``try: import torch`` had grown across the
package (module-level guards with a placeholder base class, lazy imports that
raise, lazy imports that silently return a default, lazy imports that convert to
a domain error, 64 unguarded inline imports, and two ``__import__("torch")``
calls).  Every one of them re-derived the same three questions -- is torch here,
is the process group up, how many ranks are there -- and answered them slightly
differently.

This module answers them once.  It is a leaf: it imports nothing from
``aitrainer`` and nothing from torch at module scope.

The error *domain* deliberately stays at the call site.  ``P2PError``,
``FSDPConfigurationError`` and ``ProcessGroupError`` mean different things and
should keep meaning them; only the import boilerplate is shared.
"""

from __future__ import annotations

from typing import Any


def torch_available() -> bool:
    """Whether torch can be imported, without importing it.

    Deliberately not cached at module scope: a cached answer computed during
    ``import aitrainer`` would go stale for anyone who installs or blocks torch
    afterwards, and the import-time filesystem probe is pure overhead for the
    common case where callers just try to import torch anyway.
    """
    from importlib.util import find_spec
    try:
        return find_spec("torch") is not None
    except (ImportError, ValueError):
        # A blocked or partially-initialised torch is "not available", which is
        # what every caller of this function is trying to find out.
        return False


def require_torch(message: str | None = None) -> Any:
    """Return the torch module, or raise ``ImportError`` naming the caller's need."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(message or "PyTorch is required for this operation") from exc
    return torch


def dist() -> Any | None:
    """Return ``torch.distributed``, or ``None`` when torch is unavailable."""
    try:
        from torch import distributed
    except ImportError:
        return None
    return distributed


def is_distributed() -> bool:
    """Whether a process group is actually up (not merely importable)."""
    distributed = dist()
    if distributed is None:
        return False
    try:
        return bool(distributed.is_available() and distributed.is_initialized())
    except (RuntimeError, ValueError):
        # A torn-down or half-initialized group must read as "not distributed"
        # rather than propagating out of a size query.
        return False


def world_size(group: Any = None) -> int:
    """Ranks in ``group``, or ``1`` when there is no live process group.

    Collapses the four near-identical private helpers previously defined in
    ``parallel/collectives.py``, ``parallel/tp.py``, ``parallel/sp_ulysses.py``
    and inline in ``plugins/transformer.py``.
    """
    if not is_distributed():
        return 1
    return int(dist().get_world_size(group))


def rank(group: Any = None) -> int:
    """This rank's index within ``group``, or ``0`` when there is no group."""
    if not is_distributed():
        return 0
    return int(dist().get_rank(group))
