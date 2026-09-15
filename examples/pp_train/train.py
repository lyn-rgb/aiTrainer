"""Single-stage PP schedule smoke example; multi-rank uses torchrun."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import GPipeSchedule, P2PCommunicator  # noqa: E402


def main() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    schedule = GPipeSchedule(stage_module=module,
                             communicator=P2PCommunicator(pp_rank=0, pp_size=1),
                             num_microbatches=2,
                             loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    batch = (torch.randn(4, 8), torch.randn(4, 4))
    result = schedule.run(batch)
    print(f"pp1 schedule={result.microbatches} microbatches backward_complete={result.backward_complete}")


if __name__ == "__main__":
    main()
