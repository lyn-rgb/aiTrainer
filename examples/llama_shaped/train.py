"""Train the Llama-shaped model under any of the parallel strategies.

This is the guide's worked example (``docs/使用指南.md``): one model, one
training loop, and the strategy as a flag -- so the only thing that differs
between runs is the configuration handed to ``parallelize``.

**This script runs exactly ONE rank and never starts another process.**  A
multi-rank run is started the way every other example in this repository starts
one, by launching the ranks yourself::

    # single process, no process group needed
    python examples/llama_shaped/train.py --strategy single

    # one OS process per rank, rendezvous set by hand
    for r in 0 1; do MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=$r \
        python examples/llama_shaped/train.py --strategy fsdp & done; wait

    # or, where torchrun's rendezvous works (it resolves the hostname through
    # mDNS on some hosts and dies with `gai error: 8`, hence the loop above)
    torchrun --standalone --nproc_per_node=2 \
        examples/llama_shaped/train.py --strategy fsdp

``--world`` is required for a multi-rank strategy and must equal the product of
the axes; it is a *declaration* of how many ranks you are about to start, not an
instruction to start them.  A mismatch between it and ``WORLD_SIZE`` is caught
here rather than several frames deep inside a collective.

**Why there is no built-in launcher.**  An earlier version spawned the ranks
itself.  Two things went wrong with it, and the second was severe.  Its timeout
did not kill, so a stuck rank was left behind and the next attempt added more --
they accumulated until the machine ran out of process slots.  And once it was
given a flag to tell a rank apart from the launcher, that flag guarded only the
leftover check and not the launching itself: each child recomputed the world
size from its inherited ``--strategy`` and spawned its own children, doubling
per generation.  That is a fork bomb, and it cost a reboot to clear.

A single-rank script cannot do either.  Everything that spawns lives in
``tests/multirank/harness.py``, which has a hard deadline and kills on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (dp, tp, pp, sp) per strategy.  The product must equal WORLD_SIZE, because a
# configuration whose axes do not multiply to the world size is rejected by
# config validation rather than approximated.
STRATEGIES: dict[str, dict] = {
    "single": {"dp": 1, "tp": 1, "pp": 1, "sp": False},
    "fsdp": {"dp": 2, "tp": 1, "pp": 1, "sp": False},
    "tp": {"dp": 1, "tp": 2, "pp": 1, "sp": False},
    "tp_sp": {"dp": 1, "tp": 2, "pp": 1, "sp": True},
    "pp": {"dp": 1, "tp": 1, "pp": 2, "sp": False},
    "dp2_tp2_sp": {"dp": 2, "tp": 2, "pp": 1, "sp": True},
}
DEFAULT_WORLD = {"single": 1, "fsdp": 2, "tp": 2, "tp_sp": 2, "pp": 2, "dp2_tp2_sp": 4}


def build_config(strategy: str):
    from aitrainer import FrameworkConfig

    shape = STRATEGIES[strategy]
    values: dict = {"parallel": {"dp_size": shape["dp"], "tp_size": shape["tp"],
                                 "pp_size": shape["pp"]}}
    if shape["sp"]:
        values["parallel"]["sp_backend"] = "megatron"
    if shape["pp"] > 1:
        # gpipe is the default schedule; 1f1b needs num_microbatches >= pp_size
        # and is exercised by ``--schedule 1f1b``.
        values["parallel"]["pp_schedule"] = "gpipe"
        values["parallel"]["num_microbatches"] = shape["pp"]
    if shape["dp"] > 1:
        values["fsdp"] = {"enabled": True}
    return FrameworkConfig.from_dict(values)


def run(strategy: str, *, steps: int, seed: int, batch: int, length: int,
        schedule: str, quiet: bool) -> dict:
    import model as llama
    import torch

    from aitrainer import Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    config = build_config(strategy)
    if schedule != "gpipe":
        from dataclasses import replace
        config = config.replace(parallel=replace(config.parallel, pp_schedule=schedule))

    runtime = Runtime(device=config.device, seed=seed)
    try:
        torch.manual_seed(seed)
        raw = llama.LlamaShapedForCausalLM()
        # The only model-specific thing a pipeline run needs: see
        # model.execution_order for why the split cannot infer it.
        model = parallelize(raw, config=config, runtime=runtime,
                            execution_order=llama.execution_order)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(model, optimizer, config=config, runtime=runtime,
                          loss_fn=llama.causal_lm_loss)
        data = llama.batches(seed=seed, steps=steps, batch=batch, length=length)

        started = time.perf_counter()
        history = trainer.fit(data, epochs=1)
        elapsed = time.perf_counter() - started

        inner = getattr(trainer.model, "module", trainer.model)
        # Measured, not assumed: a TP rank reports the GLOBAL count (45,344 at
        # tp=2, the same as single) because the parameters are DTensors and
        # ``numel()`` is the logical size, not the local shard's.  A PP rank
        # reports less, because a stage genuinely holds a subset of the model.
        # Reported per rank rather than summed, since comparing the strategies
        # is the reader's call and a sum would need a collective to be correct.
        local = _parameter_count(inner)
        summary = {
            "strategy": strategy, "rank": runtime.rank, "world": runtime.world_size,
            "steps": trainer.optimizer_step, "first_loss": history[0].loss,
            "last_loss": history[-1].loss, "seconds": elapsed,
            "rank_parameters": local,
            "stage": int(getattr(inner, "stage_id", 0)),
        }
        if not quiet:
            print(f"rank {runtime.rank}/{runtime.world_size} "
                  f"stage={summary['stage']} steps={summary['steps']} "
                  f"loss {summary['first_loss']:.6f} -> {summary['last_loss']:.6f} "
                  f"took {elapsed:.2f}s", flush=True)
        return summary
    finally:
        runtime.close()


def _parameter_count(module) -> int:
    """Total logical parameters this rank's model exposes.

    Two accessors, because the PP path does not wrap an ``nn.Module``:
    ``PipelineStage`` is a plain class holding ``.module``, and it exposes
    ``parameters()`` without the ``named_`` variant.  Asking only for named
    parameters raised ``AttributeError``, which an earlier version swallowed,
    and rank 0 at pp=2 printed ``rank0_params=0`` for a stage that in fact holds
    the embedding and three layers.
    """
    named = getattr(module, "named_parameters", None)
    if callable(named):
        return sum(parameter.numel() for name, parameter in named()
                   if not name.endswith("_metadata"))
    plain = getattr(module, "parameters", None)
    if callable(plain):
        return sum(parameter.numel() for parameter in plain())
    return 0


def declared_capacity(strategy: str) -> int:
    """How many ranks this strategy's axes multiply to."""
    shape = STRATEGIES[strategy]
    return shape["dp"] * shape["tp"] * shape["pp"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="single")
    parser.add_argument("--world", type=int, default=None,
                        help="how many ranks you are launching; must match the axes")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--schedule", default="gpipe", choices=("gpipe", "1f1b"))
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    needed = declared_capacity(arguments.strategy)
    if needed > 1 and arguments.world is None:
        parser.error(
            f"--strategy {arguments.strategy} needs {needed} ranks; start that many "
            f"processes yourself and pass --world {needed} to each. See this file's "
            f"docstring for the launch loop.")
    world = arguments.world or 1

    if needed != world:
        parser.error(f"--strategy {arguments.strategy} needs --world {needed}, not {world}")

    # The declaration has to match the process group, or the mismatch surfaces
    # later as a collective that never completes.
    from_env = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if from_env != world:
        print(f"refusing to start: --world {world} but WORLD_SIZE={from_env}. The "
              f"process group would be a different size than the configuration "
              f"describes. Start {world} processes with WORLD_SIZE={world}, or drop "
              f"--world for a single-process run.")
        return 1

    summary = run(arguments.strategy, steps=arguments.steps, seed=arguments.seed,
                  batch=arguments.batch, length=arguments.length,
                  schedule=arguments.schedule, quiet=arguments.quiet)
    if summary["rank"] == 0:
        print(f"{summary['strategy']}: rank0 loss {summary['first_loss']:.6f} -> "
              f"{summary['last_loss']:.6f} steps={summary['steps']} "
              f"rank0_params={summary['rank_parameters']:,} took {summary['seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
