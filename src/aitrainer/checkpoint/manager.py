"""Atomic local and DCP checkpoint orchestration.

The manager writes a temporary directory and atomically publishes a ``READY``
marker only after model, optimizer, scheduler, scaler, RNG and metadata are
complete. A reader rejects directories without that marker or with a world-size
mismatch unless the caller explicitly enables a supported resharding path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping

from .format import file_checksum


def _remove_path(path: Path) -> None:
    """Remove whatever is at ``path``: a directory tree, a file, or a symlink.

    ``shutil.rmtree`` refuses files (NotADirectoryError) and symlinks (OSError),
    so a stale single-file checkpoint or a `latest` symlink at the target path
    left a `.NAME.previous` that poisoned every later save.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


class CheckpointError(RuntimeError):
    """Base checkpoint failure with a recoverable path/context."""


class IncompleteCheckpointError(CheckpointError):
    """Raised when an atomic checkpoint was not fully published."""


class CheckpointSchemaError(CheckpointError):
    """Raised when checkpoint metadata is incompatible with the current run."""


@dataclass(frozen=True)
class CheckpointMetadata:
    format_version: int
    global_step: int
    optimizer_step: int
    world_size: int
    rank: int
    config: dict[str, Any]
    rank_mapping: dict[str, Any]
    backend: str | None
    layout_schema: str = "v1"


def _torch_rng_state() -> dict[str, Any]:
    try:
        import torch
        return {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    except ImportError as exc:
        raise CheckpointError("PyTorch is required to save RNG state") from exc


def _save_payload(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("PyTorch is required for checkpoint payloads") from exc
    torch.save(dict(payload), path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


class CheckpointManager:
    """Save/load checkpoints for a trainer or explicit state dictionaries."""

    def __init__(self, *, runtime: Any = None, allow_world_size_change: bool = False) -> None:
        self.runtime = runtime
        self.allow_world_size_change = allow_world_size_change

    def _runtime_values(self) -> tuple[int, int, str | None, int]:
        """Rank and world size, from the Runtime if given, else from the process group.

        Falling back to ``torch.distributed`` matters: a manager constructed
        without a runtime used to report rank 0 / world size 1 even inside a
        two-rank job, which wrote wrong metadata into every checkpoint and made
        any decision that depends on "am I rank 0" wrong on every rank.
        """
        state = getattr(self.runtime, "state", None)
        if state is not None:
            return (int(getattr(state, "rank", 0)), int(getattr(state, "world_size", 1)),
                    getattr(state, "backend", None), int(getattr(state, "local_rank", 0)))
        try:
            import torch.distributed as dist
        except ImportError:
            return (0, 1, None, 0)
        if dist.is_available() and dist.is_initialized():
            return (int(dist.get_rank()), int(dist.get_world_size()), dist.get_backend(), 0)
        return (0, 1, None, 0)

    def save(self, path: str | Path, *, model: Any, optimizer: Any,
             scheduler: Any = None, scaler: Any = None, global_step: int = 0,
             optimizer_step: int = 0, config: Mapping[str, Any] | None = None,
             sampler_state: Mapping[str, Any] | None = None,
             rank_mapping: Mapping[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rank, world_size, backend, _ = self._runtime_values()
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            payload = {
                "format_version": 2,
                "model": model.state_dict() if hasattr(model, "state_dict") else model,
                "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else optimizer,
                "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
                "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
                "global_step": int(global_step), "optimizer_step": int(optimizer_step),
                "rng_python": random.getstate(), "rng_torch": _torch_rng_state(),
                "sampler_state": dict(sampler_state or {}),
            }
            _save_payload(temporary / "rank_state.pt", payload)
            metadata = CheckpointMetadata(2, int(global_step), int(optimizer_step), world_size, rank,
                                          dict(config or {}), dict(rank_mapping or {}), backend)
            metadata_path = temporary / "metadata.json"
            metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with metadata_path.open("rb") as handle:
                os.fsync(handle.fileno())
            checksums = {str(item.relative_to(temporary)): file_checksum(item)
                         for item in temporary.rglob("*") if item.is_file()}
            (temporary / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n",
                                                       encoding="utf-8")
            (temporary / "READY").write_text("complete\n", encoding="utf-8")
            with (temporary / "READY").open("rb") as handle:
                os.fsync(handle.fileno())
            self._publish(temporary, target)
            return target
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @staticmethod
    def _publish(temporary: Path, target: Path) -> None:
        """Atomically move a fully-written temp directory onto ``target``.

        ``os.replace`` onto an existing NON-EMPTY directory raises OSError 66
        ("Directory not empty"), so a second save to the same path used to fail
        outright -- and the caller then deleted the freshly written checkpoint.
        The previous checkpoint is moved aside first and restored on failure.
        """
        # Follow a symlink so the common `latest -> step-N` layout keeps working;
        # replacing the link with a real directory would silently move subsequent
        # checkpoints onto local disk.
        if target.is_symlink():
            target = target.resolve()
        if not target.exists():
            os.replace(temporary, target)
            return
        backup = target.with_name(f".{target.name}.previous")
        _remove_path(backup)
        os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            os.replace(backup, target)
            raise
        _remove_path(backup)

    def _verify(self, source: Path) -> dict[str, Any]:
        if not source.is_dir() or not (source / "READY").is_file():
            raise IncompleteCheckpointError(f"checkpoint is incomplete or unpublished: {source}")
        metadata_path = source / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise IncompleteCheckpointError(f"checkpoint metadata is missing or invalid: {source}") from exc
        checksums_path = source / "checksums.json"
        if checksums_path.is_file():
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            for relative, expected in checksums.items():
                candidate = source / relative
                if not candidate.is_file() or file_checksum(candidate) != expected:
                    raise IncompleteCheckpointError(f"checkpoint checksum mismatch: {candidate}")
        return metadata

    def load(self, path: str | Path, *, model: Any, optimizer: Any,
             scheduler: Any = None, scaler: Any = None, expected_world_size: int | None = None,
             restore_rng: bool = True) -> dict[str, Any]:
        source = Path(path)
        metadata = self._verify(source)
        current_world = expected_world_size if expected_world_size is not None else self._runtime_values()[1]
        saved_world = int(metadata.get("world_size", 1))
        if saved_world != current_world and not self.allow_world_size_change:
            raise CheckpointSchemaError(
                f"checkpoint world_size={saved_world} differs from current={current_world}; "
                "explicit resharding is required")
        try:
            import torch
            payload = torch.load(source / "rank_state.pt", map_location="cpu", weights_only=False)
        except (FileNotFoundError, ImportError) as exc:
            raise CheckpointError(f"checkpoint rank_state.pt is unavailable: {source}") from exc
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if restore_rng:
            random.setstate(payload["rng_python"])
            torch.set_rng_state(payload["rng_torch"]["torch"])
            if torch.cuda.is_available() and payload["rng_torch"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(payload["rng_torch"]["cuda"])
        return {"global_step": int(payload["global_step"]),
                "optimizer_step": int(payload.get("optimizer_step", payload["global_step"])),
                "sampler_state": payload.get("sampler_state", {}), "metadata": metadata}

    def _barrier(self) -> None:
        """Synchronise every rank, when there is more than one."""
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    def save_dcp(self, path: str | Path, *, state: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> Path:
        """Atomically publish a torch.distributed.checkpoint directory.

        ``dcp.save`` is COLLECTIVE: every rank writes a different shard, and the
        coordinator additionally writes ``.metadata``, all into one directory.
        This used to give each rank its own ``mkdtemp`` staging directory and
        then let every rank ``_publish`` it over the shared target -- so the last
        rank to finish replaced the directory the others had just written into.

        Measured at world_size=2: the target ended up holding only
        ``['__1_0.distcp']``, with rank 0's shard and the ``.metadata`` file gone,
        and ``load_dcp`` then failed with ``assert metadata is not None``.  The
        single-process tests could never see it, and ``save_dcp`` had no caller in
        ``src/`` at all.

        Now all ranks stage into ONE shared directory (a fixed name, not a
        per-process one), and rank 0 publishes it after a barrier.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rank, _world_size, _backend, _ = self._runtime_values()
        staging = target.with_name(f".{target.name}.tmp-staging")
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        try:
            if rank == 0 and staging.exists():
                _remove_path(staging)
            self._barrier()
            dcp.save(dict(state), storage_writer=dcp.FileSystemWriter(str(staging / "dcp")))
            self._barrier()
            if rank == 0:
                (staging / "metadata.json").write_text(
                    json.dumps(dict(metadata or {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (staging / "READY").write_text("complete\n", encoding="utf-8")
                self._publish(staging, target)
            self._barrier()
            return target
        except Exception:
            if rank == 0 and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def load_dcp(self, path: str | Path, *, state: dict[str, Any]) -> dict[str, Any]:
        source = Path(path)
        if not (source / "READY").is_file():
            raise IncompleteCheckpointError(f"DCP checkpoint is incomplete: {source}")
        try:
            import torch.distributed.checkpoint as dcp
            dcp.load(state, storage_reader=dcp.FileSystemReader(str(source / "dcp")))
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        metadata_path = source / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return metadata
