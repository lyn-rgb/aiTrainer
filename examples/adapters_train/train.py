"""High-level adapter entry point example (requires the torch dependency)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from aitrainer import GPTAdapter, Trainer, TransformerModelConfig  # noqa: E402


def build_trainer() -> Trainer:
    adapter = GPTAdapter(TransformerModelConfig(vocab_size=256, hidden_size=64, heads=4, layers=2))
    return Trainer.from_model(adapter, optimizer_kwargs={"lr": 1e-3})


if __name__ == "__main__":
    trainer = build_trainer()
    print(f"built {trainer.model.module.__class__.__name__} on {trainer.device}")
    trainer.close()

