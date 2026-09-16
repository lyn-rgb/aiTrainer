"""TP integration tests for a torchrun CPU/Gloo or CUDA/NCCL environment."""

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, TransformerTPPlan, validate
from aitrainer.parallel.tp import TPConfigurationError, device_mesh


def test_tp_configuration_is_dp_pp_exclusive():
    config = FrameworkConfig.from_dict({"parallel": {"tp_size": 2}})
    validate(config, world_size=2)


def test_the_plan_covers_a_declared_transformer_block():
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.o_proj = torch.nn.Linear(8, 8)

    styles = TransformerTPPlan().styles(Block())
    assert sorted(styles) == ["o_proj", "q_proj"]


def test_device_mesh_rejects_a_group_it_cannot_describe():
    """Without a process group there is no mesh, and torch cannot invent one."""
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized, so the guard cannot fire")
    with pytest.raises(TPConfigurationError, match="process group"):
        device_mesh(None)
