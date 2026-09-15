"""Single-rank smoke example for the Batch 2 TP modules.

For TP>1 launch with torchrun after initializing the Runtime and constructing
the explicit TP process group; the example keeps the model intentionally tiny.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import ColumnParallelLinear, RowParallelLinear  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    first = ColumnParallelLinear(8, 16, gather_output=True)
    second = RowParallelLinear(16, 4, input_is_parallel=False)
    value = torch.randn(2, 8)
    output = second(first(value))
    loss = output.square().mean()
    loss.backward()
    print(f"tp1 output_shape={tuple(output.shape)} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
