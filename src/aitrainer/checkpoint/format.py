"""Versioned checkpoint manifest schema and atomic JSON utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ModelLoadConfig:
    """Loading policy. Only these four fields are read by ``checkpoint.reader``.

    ``cpu_buffer_bytes``, ``max_inflight_reads``, ``pin_memory`` and ``node_cache``
    used to sit here; the reader performs sequential ``torch.load`` calls, never
    consulted them, and they are gone rather than left implying the loader is
    buffered, concurrent or pinned.
    """

    mode: Literal["direct_sharded", "rank0_legacy", "convert_then_load"] = "direct_sharded"
    reader: Literal["dcp", "safetensors", "torch"] = "torch"
    verify_checksum: bool = True
    allow_world_size_change: bool = False

    def validate(self) -> None:
        if self.reader not in {"dcp", "safetensors", "torch"}:
            raise ValueError(f"model load reader={self.reader!r} is not supported")


@dataclass(frozen=True)
class ShardSpec:
    tensor_name: str
    global_shape: tuple[int, ...]
    dtype: str
    layout: str
    target_rank: tuple[int, int, int] = (0, 0, 0)  # pp, tp, dp
    source_file: str = ""
    source_key: str | None = None
    source_offset: int | None = None
    source_length: int | None = None
    local_shape: tuple[int, ...] = ()
    global_offset: tuple[int, ...] = ()
    replicated: bool = False
    transform: str | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class Manifest:
    format_version: int = 1
    model_config_hash: str = ""
    tensor_schema_hash: str = ""
    world_size_at_save: int = 1
    logical_sharding: dict[str, int] = field(default_factory=lambda: {"tp": 1, "pp": 1, "dp": 1})
    tensors: dict[str, tuple[ShardSpec, ...]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError(f"unsupported manifest format_version={self.format_version}")
        if self.world_size_at_save < 1:
            raise ValueError("manifest world_size_at_save must be positive")
        for name, shards in self.tensors.items():
            if not name or not shards:
                raise ValueError("manifest tensor names and shard lists must be non-empty")
            for shard in shards:
                if shard.tensor_name != name:
                    raise ValueError(f"manifest key {name!r} disagrees with shard tensor_name={shard.tensor_name!r}")
                if any(size < 0 for size in shard.global_shape):
                    raise ValueError(f"negative global shape for {name!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["tensors"] = {name: [asdict(shard) for shard in shards] for name, shards in self.tensors.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        def parse_shard(item: dict[str, Any]) -> ShardSpec:
            normalized = dict(item)
            for key in ("global_shape", "local_shape", "global_offset", "target_rank"):
                if key in normalized and normalized[key] is not None:
                    normalized[key] = tuple(normalized[key])
            return ShardSpec(**normalized)
        tensors = {name: tuple(parse_shard(item) for item in shards)
                   for name, shards in data.get("tensors", {}).items()}
        return cls(format_version=int(data.get("format_version", 1)),
                   model_config_hash=data.get("model_config_hash", ""),
                   tensor_schema_hash=data.get("tensor_schema_hash", ""),
                   world_size_at_save=int(data.get("world_size_at_save", 1)),
                   logical_sharding=dict(data.get("logical_sharding", {"tp": 1, "pp": 1, "dp": 1})),
                   tensors=tensors)


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> Manifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint manifest does not exist: {source}")
    try:
        manifest = Manifest.from_dict(json.loads(source.read_text(encoding="utf-8")))
        manifest.validate()
        return manifest
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"checkpoint manifest is truncated or invalid: {source}") from exc
