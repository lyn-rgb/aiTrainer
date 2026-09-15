"""Explicit process-group construction from a :class:`RankMapping`."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from ..topology import RankCoordinate, RankMapping


class ProcessGroupError(RuntimeError):
    """Raised when groups are requested before distributed initialization."""


def group_creation_plan(mapping: RankMapping) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every group to create, in the single order all ranks must agree on.

    ``dist.new_group`` has to be called by EVERY rank, in the same order, with
    the same rank list each call.  Creating only a rank's own groups made the
    per-process call counter collide in the store: one index carried groups of
    different sizes, no rank observed the expected barrier count, and every
    participant blocked for the full gloo timeout (30 minutes).  Enumerating
    every group along each axis in a fixed order gives all ranks identical call
    sequences.  Kept separate from :meth:`ProcessGroups.create` so the ordering
    invariant is testable without a live process group.
    """
    sizes = {"pp": mapping.pp_size, "dp": mapping.dp_size, "tp": mapping.tp_size}
    plan: list[tuple[str, tuple[int, ...]]] = []
    for axis in ("pp", "dp", "tp"):
        others = [name for name in ("pp", "dp", "tp") if name != axis]
        for values in product(*(range(sizes[name]) for name in others)):
            chosen = {"pp": 0, "dp": 0, "tp": 0}
            chosen.update(dict(zip(others, values)))
            chosen[axis] = 0  # group_ranks varies this axis itself
            plan.append((axis, mapping.group_ranks(axis, RankCoordinate(**chosen))))
    return tuple(plan)


@dataclass
class ProcessGroups:
    mapping: RankMapping
    pp_group: Any = None
    dp_group: Any = None
    tp_group: Any = None

    @classmethod
    def create(cls, mapping: RankMapping) -> "ProcessGroups":
        try:
            import torch.distributed as dist
        except ImportError as exc:
            if mapping.world_size > 1:
                raise ProcessGroupError("PyTorch is required for multi-rank process groups") from exc
            return cls(mapping)
        if not dist.is_initialized():
            if mapping.world_size > 1:
                raise ProcessGroupError("torch.distributed process group is not initialized")
            return cls(mapping)
        # Check the world size BEFORE the single-rank short circuit.  A 1-rank
        # mapping inside an N>1 initialized world used to return ProcessGroups with
        # pp/dp/tp = None, and torch.distributed reads a None group as ALL ranks --
        # so a "single rank" reduction silently spanned the whole job.
        initialised = dist.get_world_size()
        if mapping.world_size != initialised:
            raise ProcessGroupError(
                f"rank mapping world_size={mapping.world_size} does not match "
                f"initialized world_size={initialised}"
            )
        if mapping.world_size == 1:
            return cls(mapping)      # a single rank has no groups to shard over
        groups: dict[str, Any] = {}
        local_rank = dist.get_rank()
        for axis, rank_list in group_creation_plan(mapping):
            handle = dist.new_group(list(rank_list))
            if local_rank in rank_list:
                groups[axis] = handle
        return cls(mapping, pp_group=groups["pp"], dp_group=groups["dp"], tp_group=groups["tp"])

    def group(self, axis: str) -> Any:
        if axis not in {"pp", "dp", "tp"}:
            raise ValueError(f"unknown group axis {axis!r}")
        return getattr(self, f"{axis}_group")
