"""Fixed-shape TP/SP attention smoke example."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import UlyssesAttention  # noqa: E402


def main() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 8, 4, 8, requires_grad=True)
    output = UlyssesAttention()(q, q, q)
    output.square().mean().backward()
    print(f"tp1-sp output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
