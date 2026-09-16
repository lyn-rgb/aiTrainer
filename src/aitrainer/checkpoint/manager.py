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


def _entry_is_terminal(entry: Any) -> bool:
    """DCP's ``_is_terminal`` for one element: a mapping is never terminal."""
    if isinstance(entry, Mapping):
        return False
    if isinstance(entry, list):
        return _list_is_terminal(entry)
    return True


def _list_is_terminal(value: list) -> bool:
    """Does DCP treat this list as one value rather than as a container to descend?

    It descends only when the list holds something traversable: a nested mapping,
    a list that is itself not terminal, or a tensor.  A list of numbers -- which
    is what ``LRScheduler.state_dict()`` stores in ``base_lrs`` and ``_last_lr``
    -- is therefore a single entry, not one per element.  Mirrors ``_is_terminal``
    in ``torch/distributed/checkpoint/_traverse.py``.
    """
    import torch
    for entry in value:
        if not _entry_is_terminal(entry):
            return False
        if isinstance(entry, torch.Tensor):
            return False
    return True


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

    @staticmethod
    def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
        """The keys DCP will use for ``value``, mirroring its own traversal.

        DCP flattens nested mappings into dot-joined leaf keys, and descends into
        a list only when the list holds something traversable.  ``[0.1, 0.1]``
        is therefore ONE key, not two -- getting this wrong would make the
        coverage check below compare two different key spaces.

        This mirrors ``torch/distributed/checkpoint/_traverse.py`` rather than
        importing it (a private module).  ``test_flatten_keys_matches_torch``
        pins the two together against the installed torch, so a version that
        changed the rule fails there instead of quietly weakening the check.
        """
        from collections.abc import Mapping

        if isinstance(value, Mapping):
            keys: set[str] = set()
            for name, inner in value.items():
                keys |= CheckpointManager._flatten_keys(inner, f"{prefix}{name}.")
            return keys
        if isinstance(value, list) and not _list_is_terminal(value):
            keys = set()
            for index, item in enumerate(value):
                keys |= CheckpointManager._flatten_keys(item, f"{prefix}{index}.")
            return keys
        return {prefix.rstrip(".")}

    @staticmethod
    def _namespaces(keys: Any) -> set[str]:
        """The first dotted segment of each key: ``model.unit2.w`` -> ``model``."""
        return {key.split(".", 1)[0] for key in keys}

    def _reject_unclaimed_namespaces(self, reader: Any, state: Mapping[str, Any]) -> None:
        """Refuse a checkpoint holding a namespace the restore recipe never mentions.

        DCP fills the state dict it is handed, key by key.  An entry in the
        checkpoint with no counterpart there is not an error to DCP -- it is
        simply not read.  So a scheduler saved under ``scheduler.*`` and loaded
        against a recipe without that namespace restores everything else and
        silently drops it, which is indistinguishable from a complete restore
        until the run diverges.  (The other direction is already loud: a recipe
        key the checkpoint does not have raises "Missing key in checkpoint
        state_dict".)

        The comparison is per NAMESPACE, not per key, and that is deliberate.
        A sharded checkpoint is partitioned across ranks while
        ``read_metadata`` reports the GLOBAL key set, so the unclaimed keys on
        any one rank are mostly another rank's: a pipeline stage holds
        ``model.unit0.*`` while the metadata also lists stage 1's
        ``model.unit2.*``, and the RNG state is namespaced by coordinate.  Key
        sets therefore differ legitimately on every load and comparing them
        would refuse every working checkpoint -- measured, before this was
        narrowed: 48 spurious entries under ``dp2-pp2``.

        What the namespace view does catch is the failure that actually
        happened: a whole family (scheduler, scaler, sampler) present in the
        checkpoint and absent from the recipe.
        """
        saved = set(reader.read_metadata().state_dict_metadata)
        claimed = self._flatten_keys(state)
        orphan = sorted(self._namespaces(saved - claimed) - self._namespaces(claimed))
        if orphan:
            raise CheckpointError(
                f"this checkpoint holds {len(orphan)} entries that the state being restored "
                f"never mentions, so loading would drop them without saying so: "
                f"{', '.join(orphan)}. Restore with the same objects that wrote it "
                "(scheduler, scaler, data provider), or load the entries deliberately.")

    def dcp_keys(self, path: str | Path) -> set[str]:
        """The entry names a published DCP checkpoint holds.

        Needed before a load to decide whether an OPTIONAL namespace is present:
        DCP refuses a recipe key the checkpoint lacks ("Missing key in checkpoint
        state_dict"), so a placeholder cannot simply be added on the chance that
        it was saved.
        """
        source = Path(path)
        if not (source / "READY").is_file():
            raise IncompleteCheckpointError(f"DCP checkpoint is incomplete: {source}")
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        reader = dcp.FileSystemReader(str(source / "dcp"))
        return set(reader.read_metadata().state_dict_metadata)

    def load_dcp(self, path: str | Path, *, state: dict[str, Any]) -> dict[str, Any]:
        source = Path(path)
        if not (source / "READY").is_file():
            raise IncompleteCheckpointError(f"DCP checkpoint is incomplete: {source}")
        try:
            import torch.distributed.checkpoint as dcp
            reader = dcp.FileSystemReader(str(source / "dcp"))
            self._reject_unclaimed_namespaces(reader, state)
            dcp.load(state, storage_reader=reader)
        except ImportError as exc:
            raise CheckpointError("torch.distributed.checkpoint is unavailable") from exc
        metadata_path = source / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return metadata
