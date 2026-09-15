"""Logical ``(pp, dp, tp)`` mesh manager for the validated Batch 1 subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .topology import RankCoordinate, RankMapping, make_rank_mapping


@dataclass(frozen=True)
class MeshCoordinate:
    pp: int
    dp: int
    tp: int


class DeviceMeshManager:
    """Create a PyTorch DeviceMesh when available and always expose rank metadata."""

    def __init__(self, *, pp_size: int = 1, dp_size: int = 1, tp_size: int = 1,
                 world_size: int | None = None, global_ranks: tuple[int, ...] | None = None,
                 device_type: str = "cpu", process_group: Any = None) -> None:
        self.process_group = process_group
        if world_size is None:
            try:
                import torch.distributed as dist
                # Honour an explicitly supplied group: deriving the mesh from a
                # specific process group is the only reason to pass one, and this
                # parameter used to be accepted and then silently ignored.
                if process_group is not None:
                    world_size = dist.get_world_size(process_group)
                else:
                    world_size = dist.get_world_size() if dist.is_initialized() else 1
            except ImportError:
                world_size = 1
        self.mapping = make_rank_mapping(world_size, pp_size=pp_size, dp_size=dp_size,
                                         tp_size=tp_size, global_ranks=global_ranks)
        self.device_type = device_type
        self._mesh = None
        self._submeshes: dict[str, Any] = {}
        self._groups = None
        try:
            from torch.distributed.device_mesh import init_device_mesh
            import torch.distributed as dist
        except ImportError:
            # Metadata remains usable for CPU unit tests and pre-init diagnostics.
            return
        if not dist.is_initialized():
            return
        from .parallel.groups import ProcessGroups
        # Construction failures propagate.  Swallowing them (ProcessGroupError is
        # a RuntimeError) left _groups as None, and dist reads a None group as ALL
        # ranks -- so TP sharding would silently span ranks belonging to different
        # pipeline stages instead of failing loudly.
        self._groups = ProcessGroups.create(self.mapping)
        # PyTorch DeviceMesh currently assumes its tensor contains global ranks in
        # logical order. For a custom mapping, retain explicit process groups
        # instead of silently using wrong links.
        if self.mapping.global_ranks != tuple(range(world_size)):
            return
        try:
            self._mesh = init_device_mesh(device_type, (pp_size, dp_size, tp_size),
                                          mesh_dim_names=("pp", "dp", "tp"))
            for name in ("pp", "dp", "tp"):
                self._submeshes[name] = self._mesh[name]
        except (RuntimeError, TypeError):
            # DeviceMesh mirrors the explicit groups above and is only an
            # optimisation -- losing it must not lose the handles.
            self._mesh = None

    @property
    def mesh(self) -> Any:
        return self._mesh

    @property
    def coordinate(self) -> MeshCoordinate:
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except ImportError:
            rank = 0
        coord = self.mapping.coordinate(rank)
        return MeshCoordinate(coord.pp, coord.dp, coord.tp)

    def submesh(self, name: str) -> Any:
        if name not in {"pp", "dp", "tp"}:
            raise ValueError(f"unknown mesh dimension {name!r}")
        if name in self._submeshes:
            return self._submeshes[name]
        return self._groups.group(name) if self._groups is not None else None

    def ranks(self, name: str) -> tuple[int, ...]:
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else self.mapping.global_ranks[0]
        except ImportError:
            rank = self.mapping.global_ranks[0]
        coord = self.mapping.coordinate(rank)
        return self.mapping.group_ranks(name, coord)

    @property
    def data_parallel_group(self) -> Any:
        """Return the concrete torch ProcessGroup when one was created."""
        return self._groups.dp_group if self._groups is not None else None

    @property
    def tensor_parallel_group(self) -> Any:
        """Return the TP group for the current ``(pp, dp)`` coordinate."""
        return self._groups.tp_group if self._groups is not None else None

    @property
    def pipeline_parallel_group(self) -> Any:
        """Return the PP group for the current ``(dp, tp)`` coordinate."""
        return self._groups.pp_group if self._groups is not None else None

    @property
    def data_parallel_size(self) -> int:
        return self.mapping.dp_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.mapping.tp_size

    @property
    def pipeline_parallel_size(self) -> int:
        return self.mapping.pp_size
