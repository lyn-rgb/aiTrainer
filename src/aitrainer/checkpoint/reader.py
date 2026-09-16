"""Explicit model loader API and per-rank CPU/DCP loading paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.tensors import tensor_bytes
from .format import Manifest, ModelLoadConfig, file_checksum, load_manifest


def _assign(target: Any, value: Any, *, tensor_name: str = "") -> None:
    """Copy a shard read for this rank into ``target``, which may be sharded.

    Two shapes are legitimate and they mean different things, so they must not be
    guessed at:

    * ``value`` has the target's LOCAL shape -- the checkpoint was written with
      the same topology the model uses (the normal case: the converter was given
      the TP plan's ``partition_dims`` and the same tp/dp sizes), so the shard is
      exactly the parameter this rank holds.  Copy it.
    * ``value`` has the target's GLOBAL shape -- the checkpoint was written
      unpartitioned, so this rank read the whole tensor and has to cut its own
      piece.  Let DTensor redistribute: hand-slicing cannot do it, because the
      placements can be compound (``(_StridedShard(0, sf=2), Shard(0))`` for FSDP
      over TP) and the split order is DTensor's business.

    Anything else is a layout disagreement, and reporting it is the point.  A
    checkpoint sharded for a different topology still produces a tensor of *some*
    shape, so a silent ``redistribute`` would quietly cut the wrong thing.
    """
    import torch
    placements = getattr(target, "placements", None)
    if placements is None:
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"shape mismatch for {tensor_name}: source={tuple(value.shape)} "
                             f"target={tuple(target.shape)}")
        target.copy_(value)
        return
    local = target.to_local()
    if tuple(value.shape) == tuple(local.shape):
        with torch.no_grad():
            local.copy_(value)
        return
    if tuple(value.shape) == tuple(target.shape):
        from torch.distributed.tensor import DTensor, Replicate
        whole = DTensor.from_local(value, target.device_mesh, [Replicate()])
        piece = whole.redistribute(placements=placements).to_local()
        with torch.no_grad():
            local.copy_(piece)
        return
    raise ValueError(
        f"shape mismatch for {tensor_name}: the shard read is {tuple(value.shape)}, but this "
        f"rank's parameter is {tuple(local.shape)} locally and {tuple(target.shape)} in full. "
        "The checkpoint's topology and this model's disagree -- re-convert it for this "
        "layout (convert(..., dp_sharded=..., partition_dims=...))")


class ModelLoader:
    def __init__(self, config: ModelLoadConfig | None = None) -> None:
        self.config = config or ModelLoadConfig()
        self.config.validate()

    def read_manifest(self, path: str | Path) -> Manifest:
        return load_manifest(path)

    def read_dcp_metadata(self, path: str | Path) -> Any:
        """Read DCP's small metadata object locally; tensor payloads stay per-rank."""
        if self.config.reader != "dcp":
            raise ValueError("read_dcp_metadata requires ModelLoadConfig.reader='dcp'")
        try:
            from torch.distributed.checkpoint import FileSystemReader
        except ImportError as exc:
            raise RuntimeError("torch.distributed.checkpoint is unavailable") from exc
        try:
            return FileSystemReader(str(path)).read_metadata()
        except Exception as exc:
            raise RuntimeError(f"failed to read DCP metadata from {path}") from exc

    def load_dcp_state(self, state: dict[str, Any], path: str | Path) -> dict[str, int]:
        """Fill a rank-local state dictionary through PyTorch DCP's reader."""
        if self.config.reader != "dcp":
            raise ValueError("load_dcp_state requires ModelLoadConfig.reader='dcp'")
        try:
            import torch.distributed.checkpoint as dcp
            reader = dcp.FileSystemReader(str(path))
            dcp.load(state, storage_reader=reader)
        except ImportError as exc:
            raise RuntimeError("torch.distributed.checkpoint is unavailable") from exc
        except Exception as exc:
            raise RuntimeError(f"failed to load rank-local DCP state from {path}") from exc
        tensors = sum(1 for value in state.values() if hasattr(value, "numel"))
        bytes_read = sum(tensor_bytes(value) for value in state.values()
                         if hasattr(value, "numel"))
        return {"tensors_loaded": tensors, "bytes_read": bytes_read}

    def load_state_dict(self, path: str | Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
        if self.config.mode == "direct_sharded":
            # A .pt file is the local state-dictionary path. Distributed
            # sharded readers are deliberately rejected instead of broadcasting.
            source = Path(path)
            if source.suffix != ".pt":
                raise ValueError("direct_sharded Batch 0 reader expects a local .pt file or a future manifest")
        if self.config.mode == "rank0_legacy":
            import warnings
            warnings.warn("rank0_legacy is a compatibility path with full-model memory cost", RuntimeWarning)
        import torch
        value = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(value, dict) and "model" in value:
            return value["model"]
        if not isinstance(value, dict):
            raise TypeError("checkpoint must contain a state dictionary")
        return value

    def drain(self) -> None:
        return None

    def load_for_rank(self, manifest: Manifest | str | Path, model: Any, *,
                      pp_rank: int = 0, tp_rank: int = 0, dp_rank: int = 0,
                      world_size: int | None = None,
                      logical_sharding: dict[str, int] | None = None,
                      dp_sharded: bool | None = None,
                      base_dir: str | Path | None = None,
                      map_location: str | Any = "cpu") -> dict[str, int]:
        """Read only shards addressed to one logical rank and copy them into ``model``.

        The method supports torch-serialized logical ``(pp,tp,dp)`` shard
        dictionaries. It performs no rank-0 broadcast and returns read
        statistics for diagnostics. FSDP owns the DP storage shard after the
        TP/PP-local model has been constructed.
        """
        # A checkpoint path implies where its shards live.  A caller passing an
        # already-parsed Manifest has to say so explicitly -- otherwise relative
        # `source_file` entries resolve against the CWD and the load fails unless
        # the process happens to be running in the output directory.
        resolved_base = None if base_dir is None else Path(base_dir)
        if not isinstance(manifest, Manifest):
            manifest_path = Path(manifest)
            resolved_base = resolved_base or manifest_path.parent
            manifest = load_manifest(manifest_path)
        base_dir = Path(".") if resolved_base is None else resolved_base
        current_world_size = manifest.world_size_at_save if world_size is None else world_size
        if manifest.world_size_at_save != current_world_size and not self.config.allow_world_size_change:
            raise ValueError("checkpoint world size differs; set allow_world_size_change explicitly")
        # Comparing only the world-size TOTAL accepted a tp=2,dp=1 checkpoint into
        # a tp=1,dp=2 layout: replicated tensors are shape-identical either way, so
        # nothing downstream noticed.  Callers that know their mesh can pin it.
        # Whether the DP axis split or copied is a property of the FILE and of the
        # LOADER, and they have to agree.  A checkpoint sharded along DP handed to
        # a loader that expects copies gives every rank a fragment; a copied
        # checkpoint handed to a loader expecting slices gives every rank a model
        # it believes is 1/dp of something.  Neither raises downstream.
        declared_dp_sharded = bool(getattr(manifest, "dp_sharded", False))
        if dp_sharded is not None and bool(dp_sharded) != declared_dp_sharded:
            raise ValueError(
                f"checkpoint was converted with dp_sharded={declared_dp_sharded} but this "
                f"loader asked for dp_sharded={bool(dp_sharded)}; a DP-sharded checkpoint "
                "requires FSDP on the DP axis, and a copied one requires the opposite")
        if logical_sharding is not None:
            for axis in ("pp", "tp", "dp"):
                declared = manifest.logical_sharding.get(axis)
                wanted = logical_sharding.get(axis)
                if declared is not None and wanted is not None and int(declared) != int(wanted):
                    raise ValueError(
                        f"checkpoint was sharded with {axis}={declared} but this layout uses "
                        f"{axis}={wanted}; re-shard the checkpoint or align the mesh")
        import torch
        state = model.state_dict() if hasattr(model, "state_dict") else model
        loaded, bytes_read = 0, 0
        cache: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for tensor_name, shards in manifest.tensors.items():
                if tensor_name not in state:
                    # A per-stage (PP) model legitimately lacks the tensors owned by
                    # other stages, and one manifest may cover more than this rank
                    # needs.  Only a tensor THIS model owns must be loaded; raising
                    # here would make per-stage PP loading impossible.
                    continue
                selected = [item for item in shards if item.target_rank == (pp_rank, tp_rank, dp_rank)]
                if not selected:
                    selected = [item for item in shards if item.replicated]
                if not selected:
                    # The model OWNS this tensor but nothing addresses it, so it
                    # would keep its random initialisation while the load reported
                    # success.  (This is the case worth failing loudly on; a tensor
                    # belonging to another stage was skipped above.)
                    raise KeyError(
                        f"no shard for tensor {tensor_name!r} targets rank "
                        f"(pp={pp_rank}, tp={tp_rank}, dp={dp_rank}) and none is replicated; "
                        f"the model owns it, so it would keep its random initialisation "
                        f"(manifest logical_sharding={manifest.logical_sharding})")
                for shard in selected:
                    if not shard.source_file:
                        raise ValueError(f"manifest tensor {tensor_name!r} has no source_file")
                    source_path = Path(shard.source_file)
                    if not source_path.is_absolute():
                        source_path = base_dir / source_path
                    cache_key = str(source_path)
                    if cache_key not in cache:
                        if self.config.verify_checksum and shard.checksum:
                            actual = file_checksum(source_path)
                            if actual != shard.checksum:
                                raise ValueError(
                                    f"shard {shard.source_file!r} failed checksum verification: "
                                    f"expected {shard.checksum[:12]}..., computed {actual[:12]}...")
                        payload = torch.load(source_path, map_location=map_location, weights_only=False)
                        if not isinstance(payload, dict):
                            raise TypeError(f"shard file {str(source_path)!r} must contain a dictionary")
                        cache[cache_key] = payload
                    key = shard.source_key or tensor_name
                    if key not in cache[cache_key]:
                        raise KeyError(f"tensor {key!r} missing from shard {shard.source_file!r}")
                    value = cache[cache_key][key]
                    if shard.transform == "transpose":
                        value = value.transpose(-1, -2)
                    target = state[tensor_name]
                    _assign(target, value, tensor_name=tensor_name)
                    loaded += 1
                    bytes_read += tensor_bytes(value)
        return {"tensors_loaded": loaded, "bytes_read": bytes_read}
