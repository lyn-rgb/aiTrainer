"""Run the reproducible Batch 0 tiny Transformer example on CPU or one GPU."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import ConfigPreset, Trainer  # noqa: E402
from aitrainer.runtime import seed_everything  # noqa: E402
from model import TinyTransformer  # noqa: E402


def main() -> None:
    config = ConfigPreset.single_gpu().replace(log_interval=1, seed=123)
    # Seed BEFORE building the model.  The runtime applies config.seed when it is
    # constructed inside Trainer.__init__ -- too late for weight initialisation,
    # which would then be drawn from the ambient unseeded RNG and give different
    # weights (and a different loss) on every run of this "reproducible" example.
    # Trainer.from_model performs this ordering for you.
    seed_everything(config.seed)
    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config=config)
    generator = torch.Generator().manual_seed(config.seed)
    data = [{"input_ids": torch.randint(0, 128, (2, 16), generator=generator),
             "labels": torch.randint(0, 128, (2, 16), generator=generator)} for _ in range(4)]
    history = trainer.fit(data, epochs=1)
    checkpoint = Path("tiny-transformer.pt")
    trainer.save_checkpoint(checkpoint)
    print(f"steps={trainer.global_step} loss={history[-1].loss:.4f} checkpoint={checkpoint}")
    trainer.close()


if __name__ == "__main__":
    main()
