"""Why does a head reshape fail under TP?  Point at the exact step.

The guide's example model fails at ``Attention.forward``::

    return projection(hidden).view(shape).transpose(1, 2)
    RuntimeError: shape '[2, 8, 4, 8]' is invalid for input of size 256

``512`` is the global size and ``256`` is the local one, so the tensor being
reshaped is one rank's slice -- but *being a slice* is not the problem.  A
``DTensor`` carries its placement and can propagate a reshape through it; a
plain local tensor cannot, and by the time the error is raised the difference is
gone.  Which of the two it is decides where the fix belongs: a plain tensor
means the plan dropped the placement (framework), a DTensor means the reshape
did not propagate (model code, or this torch version).

Run it with two ranks::

    for r in 0 1; do MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 WORLD_SIZE=2 RANK=$r \\
        python scripts/diag_tp_attention.py & done; wait

It prints one line per step, per rank, and exits without training anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "llama_shaped"))

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank = dist.get_rank()

import model as llama

from aitrainer import FrameworkConfig, Runtime
from aitrainer.parallelizer import parallelize


def describe(value) -> str:
    placements = getattr(value, "placements", None)
    local = ""
    if hasattr(value, "to_local"):
        local = f" local={tuple(value.to_local().shape)}"
    return f"{type(value).__name__}{tuple(value.shape)}{local} placements={placements}"


config = FrameworkConfig.from_dict({"parallel": {"tp_size": 2}})
runtime = Runtime(device="cpu", seed=0)
raw = llama.LlamaShapedForCausalLM()
module = parallelize(raw, config=config, runtime=runtime)

attention = module.model.layers[0].self_attn
weight = dict(attention.q_proj.named_parameters()).popitem()[1]
print(f"[rank {rank}] q_proj.weight        {describe(weight)}", flush=True)

hidden = torch.randn(2, 8, 32)
print(f"[rank {rank}] input               {describe(hidden)}", flush=True)

projected = attention.q_proj(hidden)
print(f"[rank {rank}] q_proj(hidden)      {describe(projected)}", flush=True)

for label, operation in (
    ("view(2,8,4,8)", lambda t: t.view(2, 8, 4, 8)),
    ("reshape(2,8,4,8)", lambda t: t.reshape(2, 8, 4, 8)),
    ("to_local().view(2,8,2,8)", lambda t: t.to_local().view(2, 8, 2, 8)
     if hasattr(t, "to_local") else None),
):
    try:
        result = operation(projected)
        print(f"[rank {rank}] {label:24s} OK    {describe(result)}", flush=True)
    except Exception as exc:                                    # noqa: BLE001 - reported
        print(f"[rank {rank}] {label:24s} FAIL  {type(exc).__name__}: {str(exc)[:90]}",
              flush=True)

# What the plan actually installed, and what the framework's own style list says.
from aitrainer.plugins.transformer import TransformerTPPlan

styles = TransformerTPPlan().styles(module.model.layers[0])
for name in sorted(styles):
    print(f"[rank {rank}] style {name:34s} {type(styles[name]).__name__} "
          f"{getattr(styles[name], 'output_layouts', '')}", flush=True)

dist.destroy_process_group()
