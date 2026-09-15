"""Small model for the FSDP example."""

import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
