"""Hardware-neutral topology descriptions and explicit rank mappings.

The first distributed batch does not guess NVLink or NUMA links. It records a
deterministic logical mapping and leaves hardware discovery as metadata, so a
future planner can replace the mapping without changing process-group APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable


class TopologyError(ValueError):
    """Raised for an invalid rank mapping or topology description."""


@dataclass(frozen=True)
class RankCoordinate:
    pp: int
    dp: int
    tp: int


@dataclass(frozen=True)
class RankMapping:
    world_size: int
    pp_size: int
    dp_size: int
    tp_size: int
    global_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = self.pp_size * self.dp_size * self.tp_size
        if min(self.world_size, self.pp_size, self.dp_size, self.tp_size) < 1:
            raise TopologyError("world and mesh dimensions must be positive")
        if expected != self.world_size:
            raise TopologyError(f"mesh product={expected} does not match world_size={self.world_size}")
        if len(self.global_ranks) != self.world_size or set(self.global_ranks) != set(range(self.world_size)):
            raise TopologyError("global_ranks must be a permutation of [0, world_size)")

    def rank(self, coordinate: RankCoordinate) -> int:
        for value, size, name in ((coordinate.pp, self.pp_size, "pp"),
                                  (coordinate.dp, self.dp_size, "dp"),
                                  (coordinate.tp, self.tp_size, "tp")):
            if not 0 <= value < size:
                raise TopologyError(f"{name} coordinate {value} is outside [0, {size})")
        logical = (coordinate.pp * self.dp_size + coordinate.dp) * self.tp_size + coordinate.tp
        return self.global_ranks[logical]

    def coordinate(self, global_rank: int) -> RankCoordinate:
        if global_rank not in self.global_ranks:
            raise TopologyError(f"rank {global_rank} is not present in mapping")
        logical = self.global_ranks.index(global_rank)
        tp = logical % self.tp_size
        dp = (logical // self.tp_size) % self.dp_size
        pp = logical // (self.tp_size * self.dp_size)
        return RankCoordinate(pp, dp, tp)

    def group_ranks(self, axis: str, coordinate: RankCoordinate) -> tuple[int, ...]:
        if axis not in {"pp", "dp", "tp"}:
            raise TopologyError(f"unknown mesh axis {axis!r}")
        sizes = {"pp": self.pp_size, "dp": self.dp_size, "tp": self.tp_size}
        ranks = []
        for value in range(sizes[axis]):
            coords = {"pp": coordinate.pp, "dp": coordinate.dp, "tp": coordinate.tp}
            coords[axis] = value
            ranks.append(self.rank(RankCoordinate(**coords)))
        return tuple(ranks)


def make_rank_mapping(world_size: int, *, pp_size: int = 1, dp_size: int = 1,
                      tp_size: int = 1, global_ranks: Iterable[int] | None = None) -> RankMapping:
    ranks = tuple(range(world_size)) if global_ranks is None else tuple(global_ranks)
    return RankMapping(world_size, pp_size, dp_size, tp_size, ranks)


@dataclass(frozen=True)
class Topology:
    hostname: str
    world_size: int
    local_rank: int
    device_count: int
    device_names: tuple[str, ...]
    backend: str
    rank_mapping: RankMapping
    peer_access: tuple[tuple[bool, ...], ...] = ()
    numa_hint: str | None = None


def inspect_hardware(mapping: RankMapping | None = None, *, backend: str = "gloo") -> Topology:
    mapping = mapping or make_rank_mapping(int(os.environ.get("WORLD_SIZE", "1")))
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        names = tuple(torch.cuda.get_device_name(i) for i in range(count)) if count else ()
        peer_access = tuple(tuple(bool(torch.cuda.can_device_access_peer(i, j)) for j in range(count))
                            for i in range(count)) if count else ()
    except ImportError:
        count, names, peer_access = 0, (), ()
    return Topology(os.uname().nodename, mapping.world_size,
                    int(os.environ.get("LOCAL_RANK", "0")), count, names, backend, mapping,
                    peer_access=peer_access, numa_hint=os.environ.get("AITRAINER_NUMA_HINT"))
