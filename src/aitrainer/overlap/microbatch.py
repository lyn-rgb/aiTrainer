"""Experimental but complete two-microbatch interleave scheduler.

The scheduler owns both microbatch activation/RNG lifetimes and deliberately
rejects pipeline, dynamic-microbatch and activation-offload combinations until
those contracts can be proven equivalent.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.lifecycle import AsyncOp, ExecutionScheduler


@dataclass(frozen=True)
class MicrobatchResult:
    losses: tuple[Any, ...]
    normalized_loss: Any
    microbatches: int
    extra_activation_bytes: int

class MicrobatchInterleaveScheduler:
    def __init__(self, *, tp_size: int, pp_size: int = 1, activation_offload: bool = False,
                 dynamic_microbatches: bool = False, scheduler: ExecutionScheduler | None = None) -> None:
        if tp_size < 2: raise ValueError("microbatch interleave requires tp_size>1")
        if pp_size != 1: raise ValueError("microbatch interleave is incompatible with PP")
        if activation_offload: raise ValueError("microbatch interleave is incompatible with activation offload")
        if dynamic_microbatches: raise ValueError("microbatch interleave requires a static two-microbatch plan")
        self.scheduler = scheduler or ExecutionScheduler(); self._ran = False
    @staticmethod
    def _rng_state() -> Any:
        try:
            import torch
            return torch.random.get_rng_state()
        except (ImportError, RuntimeError): return None
    @staticmethod
    def _rng_restore(state: Any) -> None:
        if state is not None:
            import torch
            torch.random.set_rng_state(state)
    def run(self, microbatches: Sequence[Any], *, forward_fn: Callable[[Any], Any],
            backward_fn: Callable[[Any, Any], Any], loss_fn: Callable[[Any, Any], Any] | None = None,
            normalize_loss: Callable[[Sequence[Any]], Any] | None = None) -> MicrobatchResult:
        if len(microbatches) != 2: raise ValueError("exactly two microbatches are required")
        states = [self._rng_state(), None]; activations: list[Any] = []; losses: list[Any] = []; handles: list[AsyncOp] = []
        # Forward both microbatches before either backward, leaving collective
        # handles available for the independent compute window.
        for index, batch in enumerate(microbatches):
            if states[index] is not None: self._rng_restore(states[index])
            output = forward_fn(batch); activations.append(output)
            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], AsyncOp):
                activations[-1], handle = output; handles.append(self.scheduler.register(handle))
            losses.append(loss_fn(activations[-1], batch) if loss_fn else activations[-1])
            if index == 0: states[1] = self._rng_state()
        for handle in handles: handle.wait()
        for activation, loss in zip(activations, losses): backward_fn(activation, loss)
        self._ran = True
        normalized = normalize_loss(losses) if normalize_loss else sum(losses) / len(losses)
        extra = sum(int(getattr(item, "numel", lambda: 0)()) * int(getattr(item, "element_size", lambda: 1)()) for item in activations)
        for handle in handles: handle.release()
        return MicrobatchResult(tuple(losses), normalized, 2, extra)
    def drain(self) -> None: self.scheduler.drain()

__all__ = ["MicrobatchInterleaveScheduler", "MicrobatchResult"]
