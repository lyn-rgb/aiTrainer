"""Explicit dense checkpoint to direct-sharded manifest conversion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .format import Manifest, ShardSpec, file_checksum, write_manifest


class ConversionError(ValueError):
    """Raised when a dense tensor cannot be partitioned deterministically."""


class CheckpointConverter:
    """Convert a dense state dictionary into logical TP/PP/DP rank shards.

    TP partitioning is materialized in the manifest and replicated across DP
    ranks; FSDP subsequently owns the DP storage shard after model wrapping.
    This keeps the converter independent of private FSDP flat-parameter APIs.
    """

    def __init__(self, *, partition_dims: Mapping[str, int | None] | None = None) -> None:
        self.partition_dims = dict(partition_dims or {})

    def convert(self, source: str | Path, destination: str | Path, *, world_size: int | None = None,
                dp_size: int = 1, tp_size: int = 1, pp_size: int = 1,
                stage_assignment: Mapping[str, int] | None = None,
                model_config_hash: str = "", tensor_schema_hash: str = "") -> Manifest:
        for name, value in (("dp_size", dp_size), ("tp_size", tp_size), ("pp_size", pp_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConversionError(f"{name} must be a positive integer")
        logical_world = dp_size * tp_size * pp_size
        if world_size is None:
            world_size = logical_world
        if world_size < 1:
            raise ConversionError("world_size must be positive")
        if logical_world != world_size and (dp_size, tp_size, pp_size) != (1, 1, 1):
            raise ConversionError("world_size must equal dp_size*tp_size*pp_size")
        legacy_partition_world = (dp_size, tp_size, pp_size) == (1, 1, 1) and world_size > 1
        effective_tp = world_size if legacy_partition_world else tp_size
        stage_assignment = dict(stage_assignment or {})
        if pp_size > 1 and not stage_assignment:
            # Which tensors belong to which stage is not derivable from a dense
            # state dict, so PP conversion needs it explicitly.  Previously
            # logical_sharding recorded pp_size while every pp_rank received the
            # SAME chunk: each stage file held the whole model.
            raise ConversionError(
                "pp_size>1 requires stage_assignment mapping tensor names to stage "
                "indices; without it every stage would be written the entire state dict")

        def stage_of(name: str) -> int:
            stage = int(stage_assignment.get(name, 0))
            if not 0 <= stage < pp_size:
                raise ConversionError(
                    f"stage_assignment[{name!r}]={stage} is outside [0, {pp_size})")
            return stage
        try:
            import torch
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except ImportError as exc:
            raise ConversionError("PyTorch is required for checkpoint conversion") from exc
        if isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise ConversionError("source checkpoint must contain a state dictionary")
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        rank_payloads: list[dict[str, Any]] = [dict() for _ in range(world_size)]
        tensor_specs: dict[str, tuple[ShardSpec, ...]] = {}

        def global_rank(pp_rank: int, dp_rank: int, tp_rank: int) -> int:
            return (pp_rank * dp_size + dp_rank) * tp_size + tp_rank

        for name, value in payload.items():
            if not hasattr(value, "shape"):
                continue
            partition_dim = self.partition_dims.get(name)
            if partition_dim is None:
                specs = []
                if legacy_partition_world:
                    for rank in range(world_size):
                        rank_payloads[rank][name] = value
                        specs.append(ShardSpec(name, tuple(value.shape), str(value.dtype), "replicated",
                                               target_rank=(0, rank, 0), source_file=f"rank_{rank:05d}.pt",
                                               source_key=name, local_shape=tuple(value.shape), replicated=True))
                else:
                    # A replicated tensor is copied to every stage only when no
                    # stage assignment is given; otherwise it belongs to one stage.
                    stages = range(pp_size) if not stage_assignment else (stage_of(name),)
                    for pp_rank in stages:
                        for dp_rank in range(dp_size):
                            for tp_rank in range(tp_size):
                                rank = global_rank(pp_rank, dp_rank, tp_rank)
                                rank_payloads[rank][name] = value
                                specs.append(ShardSpec(name, tuple(value.shape), str(value.dtype), "replicated",
                                                       target_rank=(pp_rank, tp_rank, dp_rank),
                                                       source_file=f"rank_{rank:05d}.pt", source_key=name,
                                                       local_shape=tuple(value.shape), replicated=True))
                tensor_specs[name] = tuple(specs)
                continue
            if not -value.ndim <= partition_dim < value.ndim:
                raise ConversionError(f"partition dim {partition_dim} invalid for tensor {name}")
            dim = partition_dim % value.ndim
            if value.shape[dim] % effective_tp:
                raise ConversionError(f"tensor {name} dim {value.shape[dim]} is not divisible by tp_size={effective_tp}")
            chunks = value.chunk(effective_tp, dim=dim)
            specs = []
            if legacy_partition_world:
                for rank, chunk in enumerate(chunks):
                    rank_payloads[rank][name] = chunk.contiguous()
                    offset = [0] * value.ndim
                    offset[dim] = rank * chunk.shape[dim]
                    specs.append(ShardSpec(name, tuple(value.shape), str(value.dtype), "hidden_sharded",
                                           target_rank=(0, rank, 0), source_file=f"rank_{rank:05d}.pt",
                                           source_key=name, local_shape=tuple(chunk.shape),
                                           global_offset=tuple(offset)))
            else:
                stage = stage_of(name)
                for dp_rank in range(dp_size):
                    for tp_rank in range(tp_size):
                        chunk = chunks[tp_rank]
                        rank = global_rank(stage, dp_rank, tp_rank)
                        rank_payloads[rank][name] = chunk.contiguous()
                        offset = [0] * value.ndim
                        offset[dim] = tp_rank * chunk.shape[dim]
                        specs.append(ShardSpec(name, tuple(value.shape), str(value.dtype), "tp_sharded",
                                               target_rank=(stage, tp_rank, dp_rank),
                                               source_file=f"rank_{rank:05d}.pt", source_key=name,
                                               local_shape=tuple(chunk.shape),
                                               global_offset=tuple(offset)))
            tensor_specs[name] = tuple(specs)
        checksums: dict[str, str] = {}
        for rank, payload_for_rank in enumerate(rank_payloads):
            rank_path = output / f"rank_{rank:05d}.pt"
            torch.save(payload_for_rank, rank_path)
            checksums[rank_path.name] = file_checksum(rank_path)
        # Populate ShardSpec.checksum so ModelLoadConfig.verify_checksum has
        # something to verify -- the field existed but was never written.
        tensor_specs = {
            name: tuple(replace(spec, checksum=checksums.get(spec.source_file)) for spec in shards)
            for name, shards in tensor_specs.items()
        }
        manifest = Manifest(model_config_hash=model_config_hash, tensor_schema_hash=tensor_schema_hash,
                            world_size_at_save=world_size,
                            logical_sharding={"tp": effective_tp if legacy_partition_world else tp_size,
                                              "pp": pp_size, "dp": dp_size},
                            tensors=tensor_specs)
        write_manifest(manifest, output / "manifest.json")
        return manifest
