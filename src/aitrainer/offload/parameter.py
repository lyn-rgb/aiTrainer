"""Parameter fetch/release lifecycle for CPU master storage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..core.tensors import tensor_bytes
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

    @staticmethod
    def _parameter_names(module: Any) -> tuple[str, ...]:
        """Parameter names of ``module`` itself, not of its descendants.

        ``named_parameters`` recurses by default, so asking a parent for its own
        names would fold every child's in as well and each layer would be
        recorded once per ancestor it has.
        """
        named = getattr(module, "named_parameters", None)
        if not callable(named):
            return ()
        return tuple(name for name, _ in named(recurse=False))

    def _descendants(self, module: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Fully qualified descendants, depth-first, each yielded once.

        Qualified, not local: ``named_children`` calls the norm inside ``b0`` and
        the one inside ``b1`` both ``norm``, so a trace of local names cannot tell
        those two positions apart -- swapping them would leave the digest
        unchanged while the model changed.

        ``_module_ids`` exists for the shared case: a submodule used from two
        parents would otherwise be recorded twice, and a plan built from that
        trace would prefetch it twice.
        """
        for name, child in getattr(module, "named_children", lambda: ())():
            if id(child) in self._module_ids:
                continue
            self._module_ids.add(id(child))
            qualified = f"{prefix}{name}"
            yield qualified, child
            yield from self._descendants(child, f"{qualified}.")

    def register_module(self, module: Any) -> int:
        """Register every parameter under ``module``, one trace entry per child.

        The trace is what a prefetch plan is built from, so its granularity bounds
        the plan's: registering the whole model records ONE entry, and a digest
        over one entry says nothing about layer order -- which measured as a
        single ``TraceEntry(module='Blk', ...)`` for a two-layer model.

        Entries carry parameter NAMES.  They used to carry ``str(id(p))``, and an
        id is a memory address: two runs of the same model produced different
        digests (measured), so ``finalize(expected=<the warmup digest>)`` could
        never match and the prefetch gate could never open.  Names are stable
        across runs and are legible in the error a mismatch raises.
        """
        self._module_ids.add(id(module))
        count = 0
        roots = [(module.__class__.__qualname__, module)]
        roots.extend(self._descendants(module))
        for name, target in roots:
            for parameter in _parameters(target):
                before = len(self._records)
                self.register(parameter)
                count += len(self._records) - before
            self.trace.record(name, self._parameter_names(target))
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

    @staticmethod
    def _copy_event() -> Any:
        """A marker for the copies just issued, or None where that means nothing.

        On CUDA an event recorded on the current stream completes when the
        non-blocking copies ahead of it have landed.  On CPU the copies are
        already done and there is nothing to wait for, so the caller gets None
        and :meth:`_synchronize` skips it -- same code path, no branch at the
        call site.
        """
        try:
            import torch
            if torch.cuda.is_available():
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                return event
        except ImportError:                                   # pragma: no cover
            pass
        return None

    @staticmethod
    def _synchronize(pending: Any) -> None:
        for event in pending or ():
            if event is not None:
                event.synchronize()

    def _target_device(self, module: Any, device: Any | None) -> Any:
        if device is not None:
            return device
        for parameter in _parameters(module):
            return getattr(parameter, "device", None)
        return None

    def prefetch_async(self, module: Any, *, device: Any | None = None) -> Any:
        """Start staging this module NOW and return a handle to wait on.

        The point is the timing.  This used to store the fetch as a closure on
        ``op.handle`` with ``op._wait_fn = lambda fn: fn()``, so the copy ran when
        the consumer waited -- a deferred fetch, not an early one, and it hid
        nothing.  The copy is now issued here and ``wait()`` only synchronizes.

        Two ways it declines, and both fetch synchronously instead of returning
        a handle the caller would read as "staged": the trace gate is closed
        (unvalidated access order), or the coordinator refuses for budget
        (:attr:`max_prefetched_bytes`).  The second used to return None without
        fetching anything, leaving the parameters on the host with nothing
        raised.

        Contract: the returned handle must be waited before the module runs.
        ``parameter.data`` is repointed at the destination tensor immediately,
        and with ``non_blocking`` copies the contents are still in flight.
        """
        target = self._target_device(module, device)
        if str(target) == "cpu":
            # Nothing to stage: the master copy is already where it would go.
            self.fetch(module=module, device=target)
            return None
        if not self.trace.valid:
            self.fetch(module=module, device=target)
            return None
        key = "module:" + str(id(module))
        op = self.trace.fetch(key, bytes_=sum(tensor_bytes(p) for p in _parameters(module)))
        if op is None:
            self.fetch(module=module, device=target)
            return None
        pending: list[Any] = []
        for item in _parameters(module):
            record = self._record(item)
            if str(getattr(item, "device", "cpu")) == str(target):
                record.fetched = True
                record.device = target
                continue
            master = record.master
            # Pinning is what makes this genuinely asynchronous: a pageable
            # source makes torch fall back to a blocking copy (correct, just not
            # overlapped).  The master comes from the pinned pool when one is
            # configured, so whether this overlaps is decided at registration.
            item.data = master.to(device=target, non_blocking=True)
            pending.append(self._copy_event())
            record.fetched = True; record.device = target
        op.handle = pending
        op._wait_fn = self._synchronize
        return op

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
        cpu_bytes = sum(tensor_bytes(r.master) for r in self._records.values())
        return {"parameters": len(self._records), "cpu_bytes": cpu_bytes,
                "fetched": sum(r.fetched for r in self._records.values())}

    def clear(self) -> None:
        if any(record.fetched for record in self._records.values()):
            raise ParameterOffloadError("cannot clear while parameters are fetched")
        if self.pool is not None:
            for record in self._records.values():
                self.pool.release(record.master)
        self._records.clear(); self._module_ids.clear()
