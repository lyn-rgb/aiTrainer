"""Tiny Transformer used by the Batch 0 smoke example."""

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, heads: int = 4,
                 layers: int = 2, max_length: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        block = nn.TransformerEncoderLayer(hidden_size, heads, 4 * hidden_size,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        hidden = self.encoder(hidden)
        logits = self.lm_head(self.norm(hidden))
        if labels is None:
            return {"logits": logits}
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"loss": loss, "logits": logits}
