"""Read-only validation and human-readable environment diagnostics."""

from __future__ import annotations

import platform
import os
from pathlib import Path
from typing import Any

from .capability import capability_matrix, installed_optional_backends, validate_capabilities
from .config import FrameworkConfig
from .topology import inspect_hardware, make_rank_mapping


def diagnose_exception(exc: BaseException, *, rank: int = 0, group: str | None = None,
                       stage: int | None = None, microbatch: int | None = None,
                       module: str | None = None, tensor: Any = None) -> dict[str, Any]:
    message = str(exc)
    kind = "unknown"
    suggestion = "inspect the chained exception and rank logs"
    lowered = message.lower()
    if "timeout" in lowered or "nccl" in lowered or "gloo" in lowered:
        kind, suggestion = "communication", "check peer ranks, process-group membership, and collective order"
    elif "out of memory" in lowered or "cuda error" in lowered:
        kind, suggestion = "oom", "reduce micro-batch size or activation memory and inspect the memory planner"
    elif "checkpoint" in lowered or "manifest" in lowered:
        kind, suggestion = "checkpoint", "check READY marker, checksums, schema, and world-size compatibility"
    return {"kind": kind, "message": message, "rank": rank, "group": group, "stage": stage,
            "microbatch": microbatch, "module": module, "tensor": tensor, "suggestion": suggestion}


def inspect_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"status": "missing", "path": str(source)}
    if source.is_dir() and not (source / "READY").is_file():
        return {"status": "incomplete", "path": str(source), "missing": "READY"}
    return {"status": "ready", "path": str(source), "has_manifest": (source / "manifest.json").is_file(),
            "has_metadata": (source / "metadata.json").is_file()}


def validate(config: FrameworkConfig, *, world_size: int = 1) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    validate_capabilities(config, world_size=world_size, torch_module=torch)


def inspect_topology() -> dict[str, Any]:
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
        version = torch.__version__
    except ImportError:
        cuda_available, device_count, devices, version = False, 0, [], None
    topology = inspect_hardware(make_rank_mapping(int(os.environ.get("WORLD_SIZE", "1"))))
    return {"python": platform.python_version(), "torch": version,
            "cuda_available": cuda_available, "device_count": device_count, "devices": devices,
            "world_size": topology.world_size,
            "rank_mapping": {"global_ranks": list(topology.rank_mapping.global_ranks),
                              "pp_size": topology.rank_mapping.pp_size,
                              "dp_size": topology.rank_mapping.dp_size,
                              "tp_size": topology.rank_mapping.tp_size},
            "hostname": topology.hostname, "backend": topology.backend,
            "peer_access": [list(row) for row in topology.peer_access],
            "numa_hint": topology.numa_hint,
            "optional_backends": installed_optional_backends()}


def dry_run(config: FrameworkConfig | None = None, *, world_size: int | None = None) -> dict[str, Any]:
    config = config or FrameworkConfig()
    # Validate against the world the config describes.  Defaulting to 1 made
    # every distributed config report a spurious product mismatch.
    if world_size is None:
        world_size = config.parallel.dp_size * config.parallel.tp_size * config.parallel.pp_size
    validate(config, world_size=world_size)
    return {"status": "ok", "world_size": world_size, "config": config.to_dict(),
            "capabilities": [{"name": c.name, "status": c.status.value, "reason": c.reason}
                             for c in capability_matrix()]}
