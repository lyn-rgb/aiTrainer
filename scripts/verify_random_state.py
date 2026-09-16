#!/usr/bin/env python3
"""Multi-rank audit of random-state consistency across the sharding strategies.

Random state is where the compound parallel axes fail quietly.  Nothing crashes
when two ranks disagree about a weight initialisation, a dropout mask or a data
order -- the run simply trains something other than what a single process would
have trained, and the loss curve looks plausible for a long time before it does
not.

Run with an explicit rendezvous (torchrun cannot start on this host -- its
rendezvous resolves the hostname through mDNS and dies with ``gai error: 8``)::

    for r in 0 1; do MASTER_ADDR=127.0.0.1 MASTER_PORT=29860 WORLD_SIZE=2 RANK=$r \\
        PYTHONPATH=src python scripts/verify_random_state.py & done; wait

Every check prints one line per rank plus a verdict, so a failure names the
axis and the rank rather than "the run diverged".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHECKS: dict = {}


def check(name: str):
    def register(function):
        CHECKS[name] = function
        return function
    return register


class Report:
    """Collects per-rank observations and renders one verdict per check."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.rows: list[tuple[str, str, str]] = []

    def say(self, check: str, verdict: str, detail: str = "") -> None:
        self.rows.append((check, verdict, detail))
        marker = {"ok": "  ok  ", "FAIL": " FAIL ", "note": " note "}[verdict]
        print(f"[rank {self.rank}]{marker}{check}"
              + (f"  {detail}" if detail else ""), flush=True)


def _cfg(**parallel):
    from aitrainer import FrameworkConfig
    return FrameworkConfig.from_dict({"parallel": parallel})


# --------------------------------------------------------------------------- #
# 1. weight initialisation, per axis combination
# --------------------------------------------------------------------------- #
@check("init")
def initialisation(rank: int, world: int, dist, report: Report, options: dict) -> None:
    """Every rank must build the SAME model before parallelising it.

    This was broken until recently: ``Runtime`` seeded ``seed + rank``, so the
    "same" model differed per rank and only tensor parallelism hid it (its
    ``distribute_tensor`` broadcast overwrote the difference).  A pipeline split
    has no such broadcast, which is where it showed up.
    """
    import torch
    from torch import nn

    from aitrainer import Runtime

    class Block(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.first = nn.Linear(hidden, hidden)
            self.second = nn.Linear(hidden, hidden)

        def forward(self, value):
            return self.second(torch.relu(self.first(value)))

    runtime = Runtime(device="cpu", init_process_group=False, seed=1234)
    torch.manual_seed(1234)
    model = Block()
    state = {name: value.detach().flatten().tolist() for name, value in model.state_dict().items()}
    gathered: list = [None] * world
    dist.all_gather_object(gathered, {"seed": torch.initial_seed(), "state": state})
    if rank == 0:
        seeds = {item["seed"] for item in gathered}
        names = list(gathered[0]["state"])
        worst = 0.0
        for item in gathered[1:]:
            for name in names:
                worst = max(worst, max(abs(a - b) for a, b in
                                       zip(gathered[0]["state"][name], item["state"][name])))
        if len(seeds) != 1:
            report.say("init", "FAIL", f"ranks disagree on the global seed: {sorted(seeds)}")
        elif worst != 0.0:
            report.say("init", "FAIL", f"the same model differs across ranks by {worst:.3e}")
        else:
            report.say("init", "ok", f"seed={seeds.pop()} identical weights on all ranks")
    del runtime


# --------------------------------------------------------------------------- #
# 2. does the RNG stay in step across a forward pass?
# --------------------------------------------------------------------------- #
@check("rng-sync")
def rng_stays_in_step(rank: int, world: int, dist, report: Report, options: dict) -> None:
    """After a forward with randomness, do the ranks still hold the same RNG state?

    Tensor parallelism splits one logical tensor, so every rank must draw the
    same mask for the positions it owns -- if the ranks drift, the shards of one
    activation were dropped out differently and the model computes something no
    single process would.

    Sequence parallelism is the interesting one: the ranks own *different
    positions* of the sequence, so masks that march in lockstep are wrong in a
    different way -- see the dropout check.
    """
    import torch
    from torch import nn

    from aitrainer import Runtime

    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5), nn.Linear(8, 8))
    runtime = Runtime(device="cpu", init_process_group=False, seed=7)
    before = torch.get_rng_state()
    model(torch.randn(4, 8))
    after = torch.get_rng_state()
    gathered: list = [None] * world
    dist.all_gather_object(gathered, {"before": before.tolist(), "after": after.tolist()})
    if rank == 0:
        same_before = all(item["before"] == gathered[0]["before"] for item in gathered)
        same_after = all(item["after"] == gathered[0]["after"] for item in gathered)
        if not same_before:
            report.say("rng-sync", "FAIL", "ranks did not even start from the same RNG state")
        elif not same_after:
            report.say("rng-sync", "FAIL",
                       "a forward pass left the ranks with different RNG state -- the ranks "
                       "consumed different amounts of randomness")
        else:
            report.say("rng-sync", "ok", "identical RNG state before and after a forward")
    del runtime


# --------------------------------------------------------------------------- #
# 3. sequence parallelism vs a single process, with dropout inside the region
# --------------------------------------------------------------------------- #
@check("sp-dropout")
def sequence_parallel_dropout(rank: int, world: int, dist, report: Report, options: dict) -> None:
    """Does a sequence-parallel region reproduce a single process EXACTLY?

    Both runs see the same seed and the same sequence; the SP run splits it
    across ranks.  A masked position must be masked to the same value in both,
    which requires each rank to draw the masks belonging to ITS positions.
    Drawing from the start of the stream on every rank -- what a globally seeded
    RNG does by default -- gives every rank the masks of the first block, so the
    second half of the sequence is dropped out with the first half's mask.  The
    output differs from one process, and nothing raises.
    """
    import torch
    from torch import nn

    from aitrainer import Runtime
    from aitrainer.parallelizer import parallelize

    class Block(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.act = nn.Dropout(p=0.5)
            self.q_proj = nn.Linear(hidden, hidden)
            self.o_proj = nn.Linear(hidden, hidden)

        def forward(self, value):
            value = self.act(self.norm(value))
            return self.o_proj(torch.relu(self.q_proj(value)))

    length, seed = 8, 4242

    # Generate the input AGAIN for the second run rather than reusing `value`.
    # Reusing it leaves the two runs at different points in the stream -- the
    # first consumed the sequence when it drew the input -- so the dropout masks
    # differ and the check reports a discrepancy that is the harness's, not the
    # framework's.  (That is exactly what happened: 9.6e-01 against a real
    # 5.9e-08, and the false result survived several rounds of "investigation"
    # before the numbers were compared segment by segment.)
    torch.manual_seed(999)
    dense = Block()
    torch.manual_seed(seed)
    value = torch.randn(1, length, 8)
    expected = dense(value).detach()

    torch.manual_seed(999)
    config = _cfg(tp_size=world, sp_backend="megatron")
    runtime = Runtime(device="cpu", init_process_group=False, seed=999)
    model = parallelize(Block(), config=config, runtime=runtime)
    torch.manual_seed(seed)
    value = torch.randn(1, length, 8)
    got = model(value)
    got = got.to_local() if hasattr(got, "to_local") else got
    error = float((got.detach() - expected).abs().max())
    # A tolerance, not == 0.0: the two runs take different routes through the
    # layout machinery, so the last bits differ.  What would NOT be within
    # tolerance is a mask drawn for the wrong positions.
    report.say("sp-dropout", "ok" if error < 1e-6 else "FAIL",
               f"sequence-parallel output vs one process: max|diff| = {error:.3e}"
               + ("" if error < 1e-6 else
                  "  <- the ranks are not drawing the masks for their own positions"))
    del runtime


# --------------------------------------------------------------------------- #
# 4. does each data-parallel rank get different data?
# --------------------------------------------------------------------------- #
@check("data-shard")
def data_sharding(rank: int, world: int, dist, report: Report, options: dict) -> None:
    """Does a data-parallel rank get a way to shard the dataset?

    ``Trainer.fit`` used to call ``data.train_dataloader(dp_group=None, ...)``
    unconditionally, so a provider could not tell the ranks apart and every
    rank walked the whole dataset in the same order -- data parallelism degraded
    to N replicas averaging identical gradients.  ``Trainer.dp_process_group()``
    now hands it the group built for the model.
    """
    import torch
    from torch import nn

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, value):
            return self.linear(value)

    torch.manual_seed(5)
    runtime = Runtime(device="cpu", init_process_group=False, seed=5)
    config = (FrameworkConfig.from_dict({"parallel": {"dp_size": world},
                                         "fsdp": {"enabled": True}})
              if world > 1 else FrameworkConfig.from_dict({}))
    model = parallelize(Model(), config=config, runtime=runtime) if world > 1 else Model()
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                      config=config, runtime=runtime,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    group = trainer.dp_process_group()
    report.say("data-shard", "ok" if group is not None else "FAIL",
               f"dp group handed to the provider: {'a real group' if group is not None else 'None'}"
               + ("" if group is not None else
                  "  <- every data-parallel rank would iterate the whole dataset"))
    del runtime


# --------------------------------------------------------------------------- #
# 5. does a checkpoint carry the random state?
# --------------------------------------------------------------------------- #
@check("checkpoint-rng")
def checkpoint_random_state(rank: int, world: int, dist, report: Report, options: dict) -> None:
    """Resuming has to restore the RNG, or the run is not the run it was.

    ``save_checkpoint`` stores ``torch``/``python`` RNG state and
    ``CheckpointManager.load`` restores it (``restore_rng`` defaults to True).
    The sharded path ``save_sharded``/``load_sharded`` stores only model,
    optimizer and step counters -- so a run resumed from a sharded checkpoint
    draws different randomness than the run that wrote it, and two runs of the
    same experiment diverge from there.
    """
    import random
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    from torch import nn

    from aitrainer import Runtime, Trainer

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, value):
            return self.linear(value)

    path = Path(tempfile.gettempdir()) / f"rng-audit-{os.environ.get('MASTER_PORT', '0')}"
    torch.manual_seed(3)
    runtime = Runtime(device="cpu", init_process_group=False, seed=3)
    model = Model()
    # A config that matches the world size: dp needs FSDP, everything else is fine
    # as-is.  A single-rank-invalid config fails validation before any check runs.
    # A config whose dimensions match the world, so validation passes.  TP rather
    # than DP on purpose: `dp_size>1` requires an FSDP-sharded model and is refused
    # at construction, which is a different check's subject.
    config = _cfg(tp_size=world)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                      config=config, runtime=runtime,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    torch.manual_seed(555)
    random.seed(555)
    trainer.fit([(torch.randn(2, 8), torch.randn(2, 8))], epochs=1)
    # Both streams: torch's, and Python's -- the latter advances inside `fit`
    # whenever the provider or the model draws from `random`.
    # Drawn on the training DEVICE, not on CPU.  A CPU-only draw cannot see a
    # CUDA generator: on a GPU box the sharded checkpoint restored the CPU and
    # Python streams while silently dropping the CUDA one, and this check drew
    # from torch.randn on CPU and reported success.  Picking the device makes the
    # same check cover whichever generator the run actually uses.
    device = trainer.device
    expected = [torch.randn(3, device=device).tolist(), random.random()]

    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    results = {}
    for name, save, load in (("per_rank", "save_checkpoint", "load_checkpoint"),
                             ("sharded", "save_sharded", "load_sharded")):
        torch.manual_seed(555)
        random.seed(555)
        trainer.fit([(torch.randn(2, 8), torch.randn(2, 8))], epochs=1)
        # The per-rank format writes one complete file per rank and issues no
        # collective, so each rank uses its OWN path -- two ranks sharing one
        # collide with "Directory not empty" while publishing.  The sharded
        # format is collective and requires the SAME path on every rank.  That
        # difference is the reason there are two entry points at all.
        target = path / (f"{name}_{rank}" if name == "per_rank" else name)
        getattr(trainer, save)(target)
        torch.manual_seed(991)                   # deliberately move the stream
        random.seed(991)
        getattr(trainer, load)(target)
        results[name] = [float((torch.randn(3, device=device) - torch.tensor(
            expected[0], device=device)).abs().max()),
                         abs(random.random() - expected[1])]
    if rank == 0:
        for name, label in (("per_rank", "save_checkpoint/load_checkpoint"),
                            ("sharded", "save_sharded/load_sharded")):
            distance, python_distance = results[name]
            report.say("checkpoint-rng", "ok" if distance == 0.0 and python_distance < 1e-12
                       else "FAIL",
                       f"{label} restores the stream (next-draw diff {distance:.1e} torch, "
                       f"{python_distance:.1e} python)"
                       + ("" if distance == 0.0 and python_distance < 1e-12 else
                          "  <- resuming does not reproduce the run being resumed"))
    shutil.rmtree(path, ignore_errors=True)
    del runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", default=",".join(CHECKS))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    arguments = parser.parse_args(argv)

    from datetime import timedelta

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=arguments.timeout_seconds))
    report = Report(rank)
    try:
        for name in arguments.checks.split(","):
            if name not in CHECKS:
                raise SystemExit(f"unknown check {name!r}; have {sorted(CHECKS)}")
            report.say(name, "note", "running")
            CHECKS[name](rank, world, dist, report, {})
            dist.barrier()
    finally:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        failures = [row for row in report.rows if row[1] == "FAIL"]
        print("\n=== verdict ===")
        print(json.dumps({"checks": len(report.rows), "failures": len(failures),
                          "failed": [row[0] for row in failures]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
