"""Distributed runtime ownership, device selection, seeding, and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import os
import random
from typing import Any


class RuntimeErrorBase(RuntimeError):
    """Base error for runtime startup and cleanup failures."""


class DistributedInitializationError(RuntimeErrorBase):
    """Raised when a requested process group cannot be initialized."""


@dataclass(frozen=True)
class RuntimeState:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    backend: str | None = None
    initialized_process_group: bool = False


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python and torch RNGs; callers remain responsible for data samplers."""
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class Runtime:
    """Own process-group lifecycle and rank/device state.

    ``init_process_group`` defaults to ``True`` only when torchrun advertises a
    world larger than one. A single-process invocation does not create a
    rendezvous or mutate global distributed state.
    """

    def __init__(self, *, device: str = "auto", seed: int = 42,
                 deterministic: bool = False, backend: str | None = None,
                 init_process_group: bool | None = None,
                 timeout_seconds: int = 1800) -> None:
        env = self._read_rank_env()
        selected_device = self.resolve_device(device, local_rank=env["local_rank"])
        selected_backend = backend or self.default_backend(selected_device)
        should_init = init_process_group if init_process_group is not None else env["world_size"] > 1
        owns_group = False
        if should_init and env["world_size"] > 1:
            owns_group = self._init_process_group(selected_backend, timeout_seconds)
        self.state = RuntimeState(rank=env["rank"], local_rank=env["local_rank"],
                                  world_size=env["world_size"], device=selected_device,
                                  backend=selected_backend if should_init else None,
                                  initialized_process_group=owns_group)
        self._closed = False
        self._owns_process_group = owns_group
        # NOT `seed + rank`.  The rank offset looked like the usual trick for
        # giving each rank different data, and it is the wrong tool for that
        # here: this seeds the *global* RNG, which is where a caller draws model
        # initialisation from.  So every rank built a different model.
        #
        # Measured at world=2 with `Runtime` created before the model -- which is
        # what this class's own docstring recommends and what every example in
        # this repository does -- `torch.initial_seed()` was 1234 on rank 0 and
        # 1235 on rank 1, and the "same" model differed by max|dW| = 6.9e-01
        # before a single step was taken.
        #
        # Two things hid it.  Tensor parallelism broadcasts parameters from
        # ``src_data_rank``, which overwrites the difference; FSDP's all-gather
        # reconstructs a parameter out of both ranks, so the result was a model
        # half-initialised from each -- correct-looking, and wrong.  Pipeline
        # parallelism has neither, so it is where the difference showed up.
        #
        # Data sharding is the caller's job (see ``seed_everything``), so the
        # offset bought nothing to set against that.  At rank 0 this is
        # unchanged, including every single-process run.
        seed_everything(seed, deterministic=deterministic)
        self._set_device()

    @staticmethod
    def _read_rank_env() -> dict[str, int]:
        def read(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise DistributedInitializationError(f"{name} must be an integer, got {raw!r}") from exc
            if value < 0:
                raise DistributedInitializationError(f"{name} must be >= 0, got {value}")
            return value

        rank, local_rank, world_size = read("RANK", 0), read("LOCAL_RANK", 0), read("WORLD_SIZE", 1)
        if rank >= world_size:
            raise DistributedInitializationError(f"RANK={rank} must be smaller than WORLD_SIZE={world_size}")
        return {"rank": rank, "local_rank": local_rank, "world_size": max(1, world_size)}

    @staticmethod
    def default_backend(device: str) -> str:
        return "nccl" if device.startswith("cuda") else "gloo"

    @staticmethod
    def resolve_device(device: str, *, local_rank: int = 0) -> str:
        if device != "auto":
            if device.startswith("cuda") and ":" not in device:
                return f"cuda:{local_rank}"
            return device
        try:
            import torch
            return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _init_process_group(backend: str, timeout_seconds: int) -> bool:
        try:
            import torch.distributed as dist
            if not dist.is_available():
                raise DistributedInitializationError("torch.distributed is unavailable")
            if dist.is_initialized():
                return False
            dist.init_process_group(backend=backend,
                                    timeout=datetime.timedelta(seconds=timeout_seconds))
            return True
        except DistributedInitializationError:
            raise
        except Exception as exc:
            raise DistributedInitializationError(
                f"failed to initialize process group backend={backend!r}; "
                "check torchrun rendezvous and NCCL/Gloo environment"
            ) from exc

    def _set_device(self) -> None:
        if self.state.device.startswith("cuda"):
            import torch
            torch.cuda.set_device(self.state.device)

    @property
    def rank(self) -> int:
        return self.state.rank

    @property
    def local_rank(self) -> int:
        return self.state.local_rank

    @property
    def world_size(self) -> int:
        return self.state.world_size

    @property
    def is_distributed(self) -> bool:
        return self.state.world_size > 1

    def process_group(self) -> Any:
        try:
            import torch.distributed as dist
            return dist.group.WORLD if dist.is_initialized() else None
        except ImportError:
            return None

    def drain(self) -> None:
        """Synchronize owned CUDA work before process-group destruction."""
        try:
            import torch
            if self.state.device.startswith("cuda"):
                torch.cuda.synchronize(self.state.device)
        except ImportError:
            return

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.drain()
            if self._owns_process_group:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
        finally:
            self._closed = True

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
