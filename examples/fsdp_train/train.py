"""Launch a tiny FSDP FULL_SHARD training job with torchrun.

CPU/Gloo is useful for API smoke tests; real multi-rank validation requires
the installed PyTorch build and a supported CUDA/NCCL environment.
"""

from pathlib import Path
import sys
from dataclasses import replace

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from aitrainer import ConfigPreset, DeviceMeshManager, Runtime, Trainer, wrap_fsdp  # noqa: E402
from model import TinyClassifier  # noqa: E402


def main() -> None:
    config = ConfigPreset.fsdp()
    with Runtime(device=config.device, seed=config.seed) as runtime:
        config = config.replace(parallel=replace(config.parallel, dp_size=runtime.world_size))
        mesh = DeviceMeshManager(dp_size=runtime.world_size, world_size=runtime.world_size,
                                 device_type="cuda" if runtime.state.device.startswith("cuda") else "cpu")
        module = wrap_fsdp(TinyClassifier(), runtime=runtime, mesh=mesh, config=config.fsdp)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        trainer = Trainer(module, optimizer, config=config, runtime=runtime,
                          loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1]))
        generator = torch.Generator().manual_seed(config.seed + runtime.rank)
        data = [(torch.randn(4, 8, generator=generator), torch.randint(0, 4, (4,), generator=generator))
                for _ in range(4)]
        trainer.fit(data)
        if runtime.rank == 0:
            print(f"completed {trainer.optimizer_step} optimizer steps")


if __name__ == "__main__":
    main()
