"""Single-rank smoke example for tensor parallelism.

At ``world_size == 1`` the plan is applied over a one-rank mesh, so the model
is built and run through exactly the code path a multi-rank job uses, without
needing a launcher.  For a real TP run, set ``tp_size`` to the world size and
start the process group first.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import FrameworkConfig, Runtime, parallelize  # noqa: E402


class Projector(torch.nn.Module):
    """Named projections: a TP plan addresses modules by name."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 16)
        self.o_proj = torch.nn.Linear(16, 4)

    def forward(self, value):
        return self.o_proj(torch.relu(self.q_proj(value)))


def main() -> None:
    torch.manual_seed(7)
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": 1}})
    runtime = Runtime(device="cpu", init_process_group=False, seed=7)
    model = parallelize(Projector(), config=config, runtime=runtime)
    output = model(torch.randn(2, 8))
    loss = output.square().mean()
    loss.backward()
    print(f"tp1 output_shape={tuple(output.shape)} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
