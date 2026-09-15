"""Tiny Llama/GPT/BERT-shaped adapters with explicit model contracts.

These adapters intentionally use stock PyTorch modules.  They provide a
stable structure for planning and examples; production fused kernels and
tokenizer/checkpoint formats remain separate capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    heads: int = 4
    layers: int = 2
    max_length: int = 128
    num_labels: int | None = None
    dropout: float = 0.0

    def validate(self) -> None:
        for name in ("vocab_size", "hidden_size", "heads", "layers", "max_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.hidden_size % self.heads:
            raise ValueError("model.hidden_size must be divisible by model.heads")
        if self.num_labels is not None and (not isinstance(self.num_labels, int) or self.num_labels < 1):
            raise ValueError("model.num_labels must be positive or None")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


def _build(config: TransformerModelConfig, *, causal: bool, classification: bool) -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Transformer adapters require PyTorch") from exc
    config.validate()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            layer = nn.TransformerEncoderLayer(config.hidden_size, config.heads,
                                               4 * config.hidden_size, dropout=config.dropout,
                                               batch_first=True, norm_first=True)
            self.layers = nn.ModuleList([layer if index == 0 else nn.TransformerEncoderLayer(
                config.hidden_size, config.heads, 4 * config.hidden_size,
                dropout=config.dropout, batch_first=True, norm_first=True)
                for index in range(config.layers)])
            self.norm = nn.LayerNorm(config.hidden_size)
            output_size = config.num_labels if classification and config.num_labels is not None else config.vocab_size
            self.head = nn.Linear(config.hidden_size, output_size)
            self.max_length = config.max_length
            self.causal = causal
            self.classification = classification

        def forward(self, input_ids: Any, labels: Any | None = None, attention_mask: Any | None = None) -> Any:
            hidden = self.embedding(input_ids)
            if hidden.shape[1] > self.max_length:
                raise ValueError("input sequence exceeds adapter max_length")
            causal_mask = None
            if self.causal:
                causal_mask = torch.ones((hidden.shape[1], hidden.shape[1]), device=hidden.device, dtype=torch.bool).triu(1)
            for layer in self.layers:
                hidden = layer(hidden, mask=causal_mask, src_key_padding_mask=(~attention_mask.bool() if attention_mask is not None else None))
            hidden = self.norm(hidden)
            if self.classification:
                logits = self.head(hidden[:, 0])
                result: dict[str, Any] = {"logits": logits}
                if labels is not None:
                    result["loss"] = nn.functional.cross_entropy(logits, labels)
                return result
            logits = self.head(hidden)
            result = {"logits": logits}
            if labels is not None:
                result["loss"] = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            return result

    return Model()


class TransformerAdapter:
    """Common adapter protocol for the tiny model families."""

    family = "transformer"
    causal = False
    classification = False

    def __init__(self, config: TransformerModelConfig | None = None) -> None:
        self.config = config or TransformerModelConfig()

    def build(self, *, device: Any = "cpu", dtype: Any | None = None) -> Any:
        model = _build(self.config, causal=self.causal, classification=self.classification)
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model.to(device)

    def layers(self, model: Any) -> Sequence[Any]:
        return tuple(model.layers)

    def estimate_layer_cost(self, layer: Any, input_spec: object) -> tuple[int, float, int]:
        parameters = sum(int(item.numel()) for item in layer.parameters())
        activation = int(getattr(input_spec, "numel", lambda: 0)())
        return parameters, float(parameters * 2), activation

    def split_stage(self, model: Any, *, policy: object) -> Any:
        from ..parallel.pp_shapes import split_sequential
        count = int(getattr(policy, "pp_size", policy if isinstance(policy, int) else 1))
        stages, _ = split_sequential(model.layers, count)
        return stages

    def input_spec(self) -> dict[str, Any]:
        return {"shape": (1, self.config.max_length), "dtype": "int64"}

    def output_for_loss(self, output: object, batch: object) -> Any:
        if isinstance(output, dict) and "loss" in output:
            return output["loss"]
        raise TypeError("adapter output has no loss; provide labels and a model-specific loss function")


class LlamaAdapter(TransformerAdapter):
    family = "llama"
    causal = True


class GPTAdapter(TransformerAdapter):
    family = "gpt"
    causal = True


class BERTAdapter(TransformerAdapter):
    family = "bert"
    classification = True

    def __init__(self, config: TransformerModelConfig | None = None) -> None:
        if config is None:
            config = TransformerModelConfig(num_labels=2)
        elif config.num_labels is None:
            config = TransformerModelConfig(vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                                             heads=config.heads, layers=config.layers,
                                             max_length=config.max_length, num_labels=2,
                                             dropout=config.dropout)
        super().__init__(config)


class TinyTransformerAdapter(TransformerAdapter):
    family = "tiny"
