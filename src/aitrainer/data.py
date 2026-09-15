"""Explicit data, micro-batch, variable-length and token-normalization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .parallel.pp_shapes import split_microbatches


class DataContractError(ValueError):
    """Raised when a batch violates the declared training data contract."""


# Keys that carry supervision, never model inputs.  They belong to the loss.
TARGET_KEYS: tuple[str, ...] = ("labels", "target", "targets")


def model_inputs(batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Split a batch into the args a ``forward`` should receive.

    Supervision is excluded so a model never has to accept ``labels`` just to be
    trainable: a tuple batch contributes only its first element, and a Mapping
    contributes every key except :data:`TARGET_KEYS`.  ``loss_fn`` (or the
    trainer's label fallback) still receives the whole batch.

    Edge case: a Mapping holding ONLY target keys has no inputs to forward, so it
    is passed through unchanged rather than emptied (an empty call would be worse).
    Such a batch is a caller error -- there is nothing for the model to consume.
    """
    if isinstance(batch, Mapping):
        values = {key: value for key, value in batch.items() if key not in TARGET_KEYS}
        return (), (values if values else dict(batch))
    if isinstance(batch, (tuple, list)):
        return ((batch[0],) if batch else ()), {}
    return (batch,), {}


@dataclass(frozen=True)
class BatchMetadata:
    valid_tokens: int | None = None
    seq_lens: tuple[int, ...] | None = None
    loss_normalization: str = "mean"

    def validate(self, batch_size: int | None = None) -> None:
        if self.valid_tokens is not None and self.valid_tokens < 1:
            raise DataContractError("valid_tokens must be positive")
        if self.seq_lens is not None:
            if any(length < 0 for length in self.seq_lens):
                raise DataContractError("seq_lens must be non-negative")
            if batch_size is not None and len(self.seq_lens) != batch_size:
                raise DataContractError("seq_lens length must equal batch size")
        if self.loss_normalization not in {"mean", "sum", "token_mean"}:
            raise DataContractError(f"unknown loss_normalization={self.loss_normalization!r}")


@dataclass(frozen=True)
class BatchContract:
    batch_dim: int = 0
    microbatch_count: int = 1
    variable_length: bool = False
    token_normalization: str = "mean"

    def validate(self) -> None:
        if self.batch_dim < 0 or self.microbatch_count < 1:
            raise DataContractError("batch_dim must be non-negative and microbatch_count positive")
        if self.token_normalization not in {"mean", "sum", "token_mean"}:
            raise DataContractError(f"unknown token_normalization={self.token_normalization!r}")

    def split(self, batch: Any) -> list[Any]:
        self.validate()
        parts = split_microbatches(batch, self.microbatch_count, batch_dim=self.batch_dim)
        return parts

    def normalize_loss(self, loss: Any, metadata: BatchMetadata | None = None) -> Any:
        self.validate()
        metadata = metadata or BatchMetadata(loss_normalization=self.token_normalization)
        metadata.validate()
        if not hasattr(loss, "ndim") or loss.ndim != 0:
            raise DataContractError("loss must be scalar")
        if self.token_normalization == "token_mean" or metadata.loss_normalization == "token_mean":
            if metadata.valid_tokens is None:
                raise DataContractError("token_mean requires BatchMetadata.valid_tokens")
            return loss / metadata.valid_tokens
        return loss


class StatefulSampler:
    """Deterministic sampler state wrapper for checkpoint/resume."""

    def __init__(self, sampler: Any, *, seed: int = 42) -> None:
        self.sampler = sampler
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        state = dict(self.sampler.state_dict()) if hasattr(self.sampler, "state_dict") else {}
        state.update({"epoch": self.epoch, "seed": self.seed})
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.seed = int(state.get("seed", self.seed))
        if hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(dict(state))


class DataProviderAdapter:
    """Adapter that exposes loader, epoch and sampler state without guessing."""

    def __init__(self, loader_factory: Any, *, sampler: StatefulSampler | None = None,
                 contract: BatchContract | None = None) -> None:
        self.loader_factory = loader_factory
        self.sampler = sampler
        self.contract = contract or BatchContract()
        self.contract.validate()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)

    def train_dataloader(self, *, dp_group: Any = None, seed: int = 42) -> Iterable[Any]:
        return self.loader_factory(dp_group=dp_group, seed=seed, epoch=self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "sampler": self.sampler.state_dict() if self.sampler else None}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        if self.sampler is not None and state.get("sampler") is not None:
            self.sampler.load_state_dict(state["sampler"])
