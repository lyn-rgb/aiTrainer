"""Parameter fetch/release lifecycle for CPU master storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..memory import BufferKey, PinnedBufferPool
from ..overlap.parameter import ParameterPrefetchCoordinator


class ParameterOffloadError(RuntimeError):
    """Raised for unsafe parameter registration or lifecycle transitions."""


@dataclass
class _ParameterRecord:
    parameter: Any
    master: Any
    device: Any = None
    fetched: bool = False
    # Pool identity of ``master``, kept here so ``_records`` stays keyed by
    # ``id(parameter)``.  Keying the dict by BufferKey collapsed every
    # same-shaped parameter onto one record while leaving its buffer leased.
    buffer_key: Any = None


def _parameters(module: Any) -> Iterable[Any]:
    if hasattr(module, "parameters"):
        return module.parameters()
    if isinstance(module, (list, tuple, set)):
        return (item for child in module for item in _parameters(child))
    return ()


class ParameterOffloader:
    """Keep a CPU master copy and explicitly fetch/release model parameters."""

    def __init__(self, *, pin_memory: bool = True, scheduler: Any | None = None,
                 pool: PinnedBufferPool | None = None) -> None:
        self.pin_memory = bool(pin_memory)
        self.scheduler = scheduler
        self.pool = pool
        self._records: dict[int, _ParameterRecord] = {}
        self._module_ids: set[int] = set()
        self.trace = ParameterPrefetchCoordinator(scheduler=scheduler)

    def register(self, parameter: Any) -> None:
        if parameter is None or not hasattr(parameter, "data") or not hasattr(parameter, "shape"):
            raise ParameterOffloadError("parameter must be a torch Parameter-like object")
        key = id(parameter)
        if key in self._records:
            return
        data = parameter.detach()
        if not hasattr(data, "to"):
            raise ParameterOffloadError("parameter data must support .to()")
        buffer_key = None
        if self.pool is not None:
            buffer_key = BufferKey(tuple(int(x) for x in data.shape), data.dtype)
            master = self.pool.acquire(buffer_key)
            master.copy_(data.detach().to(device="cpu"), non_blocking=False)
        else:
            master = data.detach().to(device="cpu").clone()
        self._records[key] = _ParameterRecord(parameter, master, buffer_key=buffer_key)

    def register_module(self, module: Any) -> int:
        count = 0
        for parameter in _parameters(module):
            before = len(self._records)
            self.register(parameter)
            count += len(self._records) - before
        self._module_ids.add(id(module))
        self.trace.record(module.__class__.__qualname__, tuple(str(id(p)) for p in _parameters(module)))
        return count

    def _record(self, parameter: Any) -> _ParameterRecord:
        record = self._records.get(id(parameter))
        if record is None:
            self.register(parameter)
            record = self._records[id(parameter)]
        return record

    def fetch(self, parameter: Any | None = None, module: Any | None = None,
              *, device: Any | None = None) -> int:
        if parameter is None and module is None:
            raise ParameterOffloadError("fetch requires parameter or module")
        values = [parameter] if parameter is not None else list(_parameters(module))
        moved = 0
        for item in values:
            record = self._record(item)
            target = device if device is not None else (record.device or getattr(item, "device", None))
            if target is None or str(target) == "cpu":
                record.fetched = True; record.device = target; continue
            if tuple(getattr(item, "shape", ())) != tuple(record.master.shape):
                raise ParameterOffloadError("parameter shape changed after registration")
            current = record.master.to(device=target, non_blocking=False)
            item.data = current
            record.fetched = True; record.device = target
            moved += int(getattr(current, "numel", lambda: 0)())
        return moved

    def prefetch(self, module: Any, *, device: Any | None = None) -> int:
        """Synchronously stage parameters onto the device.

        The trace gate governs only :meth:`prefetch_async`; both arms of the old
        branch called ``fetch`` identically, so the condition was inert here.
        """
        return self.fetch(module=module, device=device)

    def prefetch_async(self, module: Any, *, device: Any | None = None) -> Any:
        """Submit a trace-validated fetch while retaining a synchronous fallback."""
        if not self.trace.valid:
            self.fetch(module=module, device=device)
            return None
        def fetch_module() -> int:
            return self.fetch(module=module, device=device)
        # A module-level operation keeps the complete parameter set alive until
        # the consumer explicitly waits; the coordinator owns the handle.
        key = "module:" + str(id(module))
        op = self.trace.fetch(key, bytes_=sum(int(getattr(p, "numel", lambda: 0)()) * _itemsize(p) for p in _parameters(module)))
        if op is None:
            return None
        op.handle = fetch_module
        op._wait_fn = lambda fn: fn()
        return op

    def finalize_trace(self, expected: Any = None) -> bool:
        return self.trace.finalize(expected)

    def invalidate_trace(self) -> None:
        self.trace.invalidate()

    def release(self, parameter: Any | None = None, module: Any | None = None) -> int:
        if parameter is None and module is None:
            raise ParameterOffloadError("release requires parameter or module")
        values = [parameter] if parameter is not None else list(_parameters(module))
        moved = 0
        for item in values:
            record = self._record(item)
            if not record.fetched:
                continue
            source = item.detach()
            if tuple(getattr(source, "shape", ())) != tuple(record.master.shape):
                raise ParameterOffloadError("parameter shape changed during execution")
            if str(getattr(source, "device", "cpu")) != "cpu":
                record.master.copy_(source, non_blocking=False)
                item.data = record.master
                moved += int(getattr(source, "numel", lambda: 0)())
            record.fetched = False
        return moved

    def release_all(self, module: Any | None = None) -> int:
        if module is not None:
            return self.release(module=module)
        moved = 0
        for record in list(self._records.values()):
            moved += self.release(parameter=record.parameter)
        return moved

    def stats(self) -> dict[str, int | bool]:
        cpu_bytes = sum(int(getattr(r.master, "numel", lambda: 0)()) * _itemsize(r.master) for r in self._records.values())
        return {"parameters": len(self._records), "cpu_bytes": cpu_bytes,
                "fetched": sum(r.fetched for r in self._records.values())}

    def clear(self) -> None:
        if any(record.fetched for record in self._records.values()):
            raise ParameterOffloadError("cannot clear while parameters are fetched")
        if self.pool is not None:
            for record in self._records.values():
                self.pool.release(record.master)
        self._records.clear(); self._module_ids.clear()


def _itemsize(tensor: Any) -> int:
    value = getattr(tensor, "element_size", None)
    return int(value()) if callable(value) else 4
