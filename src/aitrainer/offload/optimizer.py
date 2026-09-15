"""CPU optimizer/state offload."""

from __future__ import annotations

from typing import Any

from ..memory import PinnedBufferPool


class OptimizerOffloadError(RuntimeError):
    """Raised when optimizer state cannot be moved safely."""


def _map_tensors(value: Any, fn: Any) -> Any:
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if hasattr(value, "is_floating_point") and hasattr(value, "to"):
        return fn(value)
    return value


class CPUOptimizerStateOffloader:
    """Move optimizer state to CPU between optimizer steps.

    The optimizer object itself is never replaced and non-tensor state is left
    untouched.  Lazy optimizers are supported: state created by ``step`` is
    moved to CPU on the first ``to_cpu`` call.
    """

    def __init__(self, *, pin_memory: bool = True, pool: PinnedBufferPool | None = None) -> None:
        self.pin_memory = bool(pin_memory)
        self.pool = pool
        self._registered: Any = None
        self._on_cpu = True
        self._reservations: dict[int, int] = {}

    def register(self, optimizer: Any) -> None:
        if optimizer is None or not hasattr(optimizer, "state"):
            raise OptimizerOffloadError("optimizer must expose a state mapping")
        self._registered = optimizer

    @property
    def registered(self) -> bool:
        return self._registered is not None

    def _optimizer(self, optimizer: Any | None) -> Any:
        result = optimizer if optimizer is not None else self._registered
        if result is None:
            raise OptimizerOffloadError("no optimizer has been registered")
        return result

    def to_device(self, optimizer: Any | None = None) -> int:
        opt = self._optimizer(optimizer)
        moved = 0
        for parameter, state in list(opt.state.items()):
            device = getattr(parameter, "device", None)
            if device is None:
                continue
            def move(tensor: Any) -> Any:
                nonlocal moved
                if getattr(tensor, "device", None) != device:
                    moved += int(getattr(tensor, "numel", lambda: 0)())
                    result = tensor.to(device=device, non_blocking=False)
                    reserved = self._reservations.pop(id(tensor), 0)
                    if self.pool is not None and reserved:
                        self.pool.release_external(reserved)
                    return result
                return tensor
            opt.state[parameter] = _map_tensors(state, move)
        self._on_cpu = False
        return moved

    def to_cpu(self, optimizer: Any | None = None) -> int:
        opt = self._optimizer(optimizer)
        moved = 0
        for parameter, state in list(opt.state.items()):
            def move(tensor: Any) -> Any:
                nonlocal moved
                if getattr(tensor, "device", None) is not None and str(tensor.device) != "cpu":
                    moved += int(getattr(tensor, "numel", lambda: 0)())
                    nbytes = int(tensor.numel()) * int(tensor.element_size())
                    if self.pool is not None:
                        self.pool.reserve_external(nbytes)
                    try:
                        result = tensor.detach().to(device="cpu", non_blocking=False)
                    except BaseException:
                        if self.pool is not None:
                            self.pool.release_external(nbytes)
                        raise
                    self._reservations[id(result)] = nbytes
                    return result
                if self.pool is not None and id(tensor) not in self._reservations:
                    nbytes = int(tensor.numel()) * int(tensor.element_size())
                    self.pool.reserve_external(nbytes)
                    self._reservations[id(tensor)] = nbytes
                return tensor
            opt.state[parameter] = _map_tensors(state, move)
        self._on_cpu = True
        return moved

    def state_stats(self, optimizer: Any | None = None) -> dict[str, int | bool]:
        opt = self._optimizer(optimizer)
        tensors = 0
        elements = 0
        devices: set[str] = set()
        def inspect(value: Any) -> Any:
            nonlocal tensors, elements
            if isinstance(value, dict):
                for item in value.values(): inspect(item)
            elif isinstance(value, (list, tuple)):
                for item in value: inspect(item)
            elif hasattr(value, "numel") and hasattr(value, "device"):
                tensors += 1; elements += int(value.numel()); devices.add(str(value.device))
            return value
        inspect(opt.state)
        return {"tensors": tensors, "elements": elements, "devices": tuple(sorted(devices)),
                "on_cpu": self._on_cpu}

    def drain(self, optimizer: Any | None = None) -> None:
        self.to_cpu(optimizer)

    def clear(self) -> None:
        if self.pool is not None:
            for nbytes in self._reservations.values():
                self.pool.release_external(nbytes)
        self._reservations.clear()
        self._registered = None
