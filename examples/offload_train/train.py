"""Minimal opt-in CPU optimizer offload example.

Parameter and activation modes are intentionally separate in this example;
the configuration rejects their unvalidated combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Same src-on-path shim the other examples use, so this runs from a bare
# checkout without `pip install -e .` (it was the only example missing it).
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import FrameworkConfig, OffloadConfig, Trainer


def build_trainer() -> Trainer:
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(),
                                torch.nn.Linear(256, 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = FrameworkConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        offload=OffloadConfig(enabled=True, optimizer=True, max_cpu_bytes=4 * 1024**3),
    )
    return Trainer(model, optimizer, config=config,
                   loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1]))


if __name__ == "__main__":
    trainer = build_trainer()
    x = torch.randn(8, 128, device=trainer.device)
    y = torch.randint(0, 10, (8,), device=trainer.device)
    print(trainer.train_step((x, y)))
    trainer.close()
