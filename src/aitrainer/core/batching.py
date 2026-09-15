"""Batch conventions: what a batch *is*, shared by every layer that reads one.

This module exists because the same four questions were answered in four places
that could disagree:

* **Which keys carry supervision?**  ``TARGET_KEYS`` was defined once in
  ``data.py`` and then re-hardcoded as a set literal four times in
  ``distributed_model.py``.  ``pp_schedule._extract_target`` did not use it at
  all and recognised only ``"labels"`` and tuple index 1.
* **What do we forward?**  The strip-set and the read-set must be the same set,
  or a model is asked for inputs it was never given.
* **How is loss computed?**  Four implementations existed: ``trainer._compute_loss``,
  two closures inside ``distributed_model``, and ``pp_schedule._loss``.
* **How do we split microbatches / normalise a loss?**  Lived in
  ``parallel/pp_shapes.py``, which forced ``data.py`` to import *upwards* into
  the parallel layer.

Moving them here puts batch conventions below both, so the dependency runs one
way and the four answers become one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class BatchContractError(ValueError):
    """Raised when a batch cannot be interpreted, or a loss is not scalar."""


# Kept as the historical name: it is raised by pipeline shape helpers too, and
# is exported from the public API under this spelling.
PipelineShapeError = BatchContractError


# Keys that carry supervision, never model inputs.  They belong to the loss.
TARGET_KEYS: tuple[str, ...] = ("labels", "target", "targets")


def strip_targets(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Drop supervision keys from a Mapping, leaving only forward inputs."""
    return {key: value for key, value in mapping.items() if key not in TARGET_KEYS}


def target_of(batch: Any) -> Any | None:
    """The supervision carried by ``batch``, or ``None`` if it carries none."""
    if isinstance(batch, Mapping):
        for key in TARGET_KEYS:
            if key in batch:
                return batch[key]
        return None
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[1]
    return None


def model_inputs(batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Split a batch into the args a ``forward`` should receive.

    Supervision is excluded so a model never has to accept ``labels`` just to be
    trainable: a tuple batch contributes only its first element, and a Mapping
    contributes every key except :data:`TARGET_KEYS`.  ``loss_fn`` (or the label
    fallback below) still receives the whole batch.

    Edge case: a Mapping holding ONLY target keys has no inputs to forward, so it
    is passed through unchanged rather than emptied (an empty call would be worse).
    Such a batch is a caller error -- there is nothing for the model to consume.
    """
    if isinstance(batch, Mapping):
        values = strip_targets(batch)
        return (), (values if values else dict(batch))
    if isinstance(batch, (tuple, list)):
        return ((batch[0],) if batch else ()), {}
    return (batch,), {}


def call_with_inputs(module: Any, batch: Any) -> Any:
    """Invoke ``module`` with a batch's forward inputs, whatever shape it has."""
    args, kwargs = model_inputs(batch)
    return module(*args, **kwargs) if args else module(**kwargs)


def forward_logits(output: Any) -> Any:
    """The tensor a loss should be computed against."""
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    return output


def cross_entropy_loss(logits: Any, target: Any) -> Any:
    """Flatten sequence dims so ``[N, T, C]`` logits pair with ``[N, T]`` labels."""
    from .torch import require_torch
    torch = require_torch("PyTorch is required to compute a loss")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def compute_loss(output: Any, batch: Any, loss_fn: Any = None) -> Any:
    """The single loss policy, in priority order.

    ``loss_fn`` wins; then a ``loss`` the model already computed (as a mapping
    entry or an attribute); then cross-entropy against the batch's target.  The
    target lookup reuses :func:`target_of`, so the strip-set above and the
    read-set here cannot drift apart -- a batch using ``"target"`` used to raise
    even though the error message promised a Mapping carrying labels.
    """
    if loss_fn is not None:
        return loss_fn(output, batch)
    if isinstance(output, Mapping) and "loss" in output:
        return output["loss"]
    if hasattr(output, "loss"):
        return output.loss
    target = target_of(batch)
    if target is None:
        raise TypeError(
            "loss_fn is required unless the model output contains 'loss' or the batch carries "
            "labels (a Mapping with 'labels', or a tuple whose second element is the target)"
        )
    return cross_entropy_loss(forward_logits(output), target)


def split_microbatches(batch: Any, count: int, *, batch_dim: int = 0) -> list[Any]:
    """Split tensor/mapping/tuple batches evenly without changing their contract."""
    if count < 1:
        raise BatchContractError("microbatch count must be positive")
    if hasattr(batch, "shape") and hasattr(batch, "split"):
        size = batch.shape[batch_dim]
        if size % count:
            raise BatchContractError(f"batch dimension {size} is not divisible by microbatches={count}")
        return list(batch.split(size // count, dim=batch_dim))
    if isinstance(batch, Mapping):
        parts = {key: split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for key, value in batch.items()}
        return [{key: values[index] for key, values in parts.items()} for index in range(count)]
    if isinstance(batch, tuple):
        parts = [split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for value in batch]
        return [tuple(values[index] for values in parts) for index in range(count)]
    if isinstance(batch, list):
        parts = [split_microbatches(value, count, batch_dim=batch_dim) if hasattr(value, "shape") else [value] * count
                 for value in batch]
        return [[values[index] for values in parts] for index in range(count)]
    raise BatchContractError(f"cannot split batch type {type(batch).__name__}")


def normalize_loss(loss: Any, *, valid_tokens: int | None = None) -> Any:
    """Normalize a scalar or token-summed loss exactly once."""
    if not hasattr(loss, "ndim") or loss.ndim != 0:
        raise BatchContractError("pipeline loss must be scalar")
    if valid_tokens is not None:
        if valid_tokens < 1:
            raise BatchContractError("valid_tokens must be positive")
        return loss / valid_tokens
    return loss


def iter_microbatches(batch: Any, count: int) -> Sequence[Any]:
    """Alias kept for readability at call sites that iterate rather than index."""
    return split_microbatches(batch, count)
