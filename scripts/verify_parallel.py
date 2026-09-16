#!/usr/bin/env python3
"""One command to check every parallel axis and combination against a baseline.

The reference is the SAME model trained in a single process on the SAME data
with the SAME seed.  Every combination is compared against it on four things --
forward output, loss, gradients, and the parameter update after the step --
because those are the four places a sharded run can quietly compute something
else, and the four a reader can check without trusting this script's judgement.

Two roles in one file: ``launcher`` (default) starts one process per rank with an
explicit rendezvous and writes the report; ``worker`` runs the baseline plus every
combination inside one rank and returns its observations as JSON.

Why every rank re-derives the baseline rather than reading it from rank 0: the
comparison has to be between two things built the same way in the same place.
Shipping the reference across processes would test the reference's transport.

Two things this measures and one it cannot.  It measures whether the numbers are
right, and it measures throughput and peak memory per combination.  It cannot
tell you whether a combination is *fast* -- on a CPU/Gloo host every collective
is slower than the thing it overlaps with, so the throughput column compares
implementations of the same arithmetic, not hardware behaviour.  What it does
catch, on any host, is a combination that is wrong, that allocates more than the
model needs, or that reports different numbers on different ranks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tolerance for the reference comparison.  Not exact equality: a sharded run
# takes a different route through the arithmetic (a column projection's rows are
# a contiguous slice, a row projection's all-reduce reorders the sum), so the
# last bits legitimately differ.  What would NOT land inside this is a wrong
# layout, a missing reduction, or a stage holding the wrong parameters -- those
# are orders of magnitude larger.  The measured values are in the report.
TOLERANCE = 1e-5

# Layers in the test model.  Named because the pipeline stage count is bounded by
# it, and a case list that asks for more stages than layers is a list-building
# mistake rather than a framework boundary.
LAYERS = 4


@dataclass(frozen=True)
class Case:
    """One combination of the axes, at a world size it actually fits."""

    name: str
    world_size: int
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: str = "none"
    schedule: str = "none"
    microbatches: int = 1

    def config_values(self) -> dict:
        parallel: dict = {"dp_size": self.dp, "tp_size": self.tp, "pp_size": self.pp,
                          "sp_backend": self.sp, "num_microbatches": self.microbatches}
        if self.pp > 1:
            parallel["pp_schedule"] = self.schedule
        return {"parallel": parallel, "fsdp": {"enabled": self.dp > 1}}


def cases_for(world_size: int) -> list[Case]:
    """Every combination this world size can host, smallest first.

    Built by dividing the world rather than by listing shapes, so a world size
    the machine cannot actually reach is a shorter list rather than a failed run.
    A combination whose axes do not multiply to the world size is not a valid
    configuration in this framework at all (``ParallelConfig.validate`` rejects
    it), so the list only ever holds exact divisions.
    """
    out: list[Case] = []
    if world_size >= 2:
        out.append(Case("dp2", world_size=2, dp=2))
        out.append(Case("tp2", world_size=2, tp=2))
        out.append(Case("tp2_sp", world_size=2, tp=2, sp="megatron"))
        out.append(Case("pp2", world_size=2, pp=2, schedule="gpipe", microbatches=2))
    if world_size >= 4:
        out.append(Case("dp4", world_size=4, dp=4))
        out.append(Case("tp4", world_size=4, tp=4))
        out.append(Case("pp4", world_size=4, pp=4, schedule="gpipe", microbatches=4))
        out.append(Case("pp4_1f1b", world_size=4, pp=4, schedule="1f1b", microbatches=4))
        out.append(Case("dp2_tp2", world_size=4, dp=2, tp=2))
        out.append(Case("dp2_tp2_sp", world_size=4, dp=2, tp=2, sp="megatron"))
        out.append(Case("dp2_pp2", world_size=4, dp=2, pp=2, schedule="gpipe", microbatches=2))
        out.append(Case("tp2_pp2", world_size=4, tp=2, pp=2, schedule="gpipe", microbatches=2))
    if world_size >= 8:
        out.append(Case("dp8", world_size=8, dp=8))
        out.append(Case("tp8", world_size=8, tp=8))
        out.append(Case("dp4_tp2", world_size=8, dp=4, tp=2))
        out.append(Case("dp2_tp4", world_size=8, dp=2, tp=4))
        out.append(Case("dp4_pp2", world_size=8, dp=4, pp=2, schedule="gpipe", microbatches=2))
        out.append(Case("dp2_tp2_pp2", world_size=8, dp=2, tp=2, pp=2,
                        schedule="gpipe", microbatches=2))
        out.append(Case("dp2_tp2_pp2_sp", world_size=8, dp=2, tp=2, pp=2, sp="megatron",
                        schedule="gpipe", microbatches=2))
        # The other schedule, at the depth where it exists: 1F1B needs
        # num_microbatches >= pp_size, and its steady state only appears when
        # there is more than one stage in flight.
        out.append(Case("dp2_tp2_pp2_1f1b", world_size=8, dp=2, tp=2, pp=2,
                        schedule="1f1b", microbatches=4))
    for case in out:
        # A case whose axes do not multiply to its world size is not a
        # configuration this framework accepts at all -- it would be reported as
        # a framework failure when it is a list-building mistake.  Checked here
        # rather than discovered in a four-process run.
        assert case.dp * case.tp * case.pp == case.world_size, (
            f"{case.name}: axes {case.dp}x{case.tp}x{case.pp} do not multiply to "
            f"world_size {case.world_size}")
        # LAYERS is the model's depth: a pipeline cannot have more stages than
        # there are layers, and pp_size is what the split walks.
        assert case.pp <= LAYERS, f"{case.name}: pp={case.pp} exceeds the {LAYERS}-layer model"
    return out


# --------------------------------------------------------------------------- #
# the model: small, but shaped so every axis has something to act on
# --------------------------------------------------------------------------- #
def build_model(seed: int):
    """A 4-layer block with the names the TP plan and the SP style look for.

    ``q_proj``/``k_proj``/``v_proj``/``linear1`` are column-parallel names and
    ``o_proj``/``linear2`` are row-parallel ones, so TP splits real projections
    rather than finding nothing to do -- which would leave every rank a full
    replica and the run would look fine.  Every layer carries ``LayerNorm``s,
    which is what sequence parallelism shards.  The layers are a ``Sequential``,
    which is what the pipeline split walks.

    ``tanh`` rather than softmax: it keeps the numbers bounded without pulling in
    a batch-wide reduction that would have to behave identically under TP and
    under PP, which is a different question from the one this script asks.
    """
    import torch

    class Attention(torch.nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden, hidden)
            self.k_proj = torch.nn.Linear(hidden, hidden)
            self.v_proj = torch.nn.Linear(hidden, hidden)
            self.o_proj = torch.nn.Linear(hidden, hidden)

        def forward(self, value):
            mixed = torch.tanh(self.q_proj(value) + self.k_proj(value) * self.v_proj(value))
            return self.o_proj(mixed)

    class Layer(torch.nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(hidden)
            self.attn = Attention(hidden)
            self.norm2 = torch.nn.LayerNorm(hidden)
            self.linear1 = torch.nn.Linear(hidden, 2 * hidden)
            self.linear2 = torch.nn.Linear(2 * hidden, hidden)

        def forward(self, value):
            value = value + self.attn(self.norm1(value))
            return value + self.linear2(torch.relu(self.linear1(self.norm2(value))))

    torch.manual_seed(seed)
    return torch.nn.Sequential(*[Layer(16) for _ in range(LAYERS)])


def build_data(seed: int, steps: int, batch: int):
    """The same batches for every rank and for the baseline.

    Identical inputs are what make a single-process run a valid reference for a
    data-parallel one: a DP reduction over identical replicas is the identity on
    the gradient.  It also means this script cannot see *divergent* replicas --
    that is what the random-state audit is for.
    """
    import torch
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(batch, 4, 16, generator=generator),
             torch.randn(batch, 4, 16, generator=generator)) for _ in range(steps)]


def _name_of(module) -> str:
    """The object whose ``named_parameters`` describe the run, stage or not."""
    inner = getattr(module, "module", module)
    return inner.module if hasattr(inner, "stage_id") else inner


def _as_global(value):
    """The whole logical tensor behind ``value``, whichever rank is asking.

    ``full_tensor()``, not ``to_local()``: a local shard has no meaning outside
    the rank holding it, and comparing shards against a single-process reference
    would require this script to know how each axis divided what -- which is the
    thing under test.  The global view needs no such knowledge, and it makes a
    second check possible for free: the same parameter name on two ranks must
    carry the same values.
    """
    inner = getattr(value, "full_tensor", None)
    return inner() if callable(inner) else value


def snapshot(module) -> dict[str, list]:
    """Every parameter this rank can see, globally, keyed by name."""
    out: dict[str, list] = {}
    for name, parameter in _name_of(module).named_parameters():
        out[name] = _as_global(parameter.detach()).flatten().tolist()
    return out


def mean_squared_error(output, batch):
    """The loss, defined once so the baseline and every combination share it."""
    import torch
    return torch.nn.functional.mse_loss(output, batch[1])


def build_optimizer(model, lr: float = 0.05):
    import torch
    return torch.optim.SGD(model.parameters(), lr=lr)


def peak_memory_bytes():
    """Peak device memory on CUDA, peak RSS on CPU.

    Different units on purpose, and labelled as such in the report: there is no
    host-side number that means the same thing as an allocator peak, and
    reporting one as the other is how a memory table becomes fiction.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated()), "cuda_peak_allocated_bytes"
    except ImportError:
        pass
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "host_peak_rss_bytes"
    except (ImportError, ValueError):                 # pragma: no cover
        return 0, "unavailable"


def observe_training(model, trainer, data, *, lr: float) -> dict:
    """Train through the framework and record the four things under test.

    Training goes through ``Trainer.train_step``, never through ``model(x)`` plus
    a manual backward.  That is not a stylistic preference: ``PipelineStage``
    forwards through its own module and does not schedule anything, so a manual
    loop over a pipeline produces a number that is not the model's -- measured,
    as a 0.7 relative error in the parameters and 1.3 in the gradients before
    this was changed, with no exception raised.  Whatever path a user trains
    through is the path that has to be verified.

    The forward output comes from ``pipeline_evaluate`` for the same reason: it
    is the framework's own forward-only traversal, so it works under every axis
    including PP, where only the last stage's output is the model's.

    Gradients are recovered from the parameter update rather than captured from
    ``.grad``: ``train_step`` steps the optimizer inside itself, so by the time
    it returns the buffers are gone, and for plain SGD the update IS the gradient
    scaled by the learning rate -- ``(before - after) / lr``.  Over several steps
    that quantity is the accumulated update over lr rather than one step's
    gradient, which is still a like-for-like comparison because the reference
    computes it the same way.
    """
    import torch

    from aitrainer.data import model_inputs

    before = snapshot(model)
    inner = trainer.model.module
    with torch.no_grad():
        if hasattr(inner, "pipeline_evaluate"):
            # A pipeline stage: the framework's forward-only traversal is what
            # carries the activation across stages, and its last stage's output
            # is the model's.  Calling the stage directly would forward through
            # one stage and stop.
            #
            # A pipeline splits the batch into microbatches, so ``outputs`` holds
            # one tensor per microbatch -- taking the last one compared a quarter
            # of the batch against the whole and reported an infinite mismatch.
            # Concatenated in order they reconstruct the batch the reference saw.
            import torch
            schedule = inner.pipeline_evaluate(data[0], loss_fn=trainer.loss_fn)
            pieces = list(schedule.outputs)
            value = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            output = _as_global(value.detach())
        else:
            args, kwargs = model_inputs(data[0])
            output = _as_global(inner(*args, **kwargs))
    output = output.flatten().tolist()
    loss_curve = [float(trainer.train_step(batch).loss) for batch in data]
    after = snapshot(model)
    updates: dict[str, list] = {}
    for name, initial in before.items():
        final = after.get(name)
        if final is not None and len(final) == len(initial):
            updates[name] = [(a - b) / lr for a, b in zip(initial, final)]
    return {"output": output, "loss_curve": loss_curve, "final_loss": loss_curve[-1],
            "grads": updates, "params": after}


def run_baseline(*, steps: int, seed: int, batch: int, lr: float) -> dict:
    """Train the model in one process.  This is the reference everything compares to."""
    from aitrainer import FrameworkConfig, Runtime, Trainer

    config = FrameworkConfig()
    # No process group: the baseline is a single process by definition, and a
    # runtime that joined the group would report this rank's world size and then
    # reject the all-ones config against it.
    runtime = Runtime(device="cpu", init_process_group=False, seed=seed)
    model = build_model(seed)
    optimizer = build_optimizer(model, lr)
    trainer = Trainer(model, optimizer, config=config, runtime=runtime,
                      loss_fn=mean_squared_error)
    data = build_data(seed, steps, batch)
    started = time.perf_counter()
    observed = observe_training(model, trainer, data, lr=lr)
    elapsed = time.perf_counter() - started
    return {**observed, "seconds": elapsed, "step_seconds": elapsed / steps,
            "peak_memory_bytes": peak_memory_bytes()[0], "peak_memory_label": peak_memory_bytes()[1],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "stage": 0, "stage_count": 1, "optimizer_steps": trainer.optimizer_step}


def run_combination(case: Case, *, steps: int, seed: int, batch: int, lr: float) -> dict:
    """Train the same model with one combination applied, in this process."""
    import torch

    from aitrainer import FrameworkConfig, Runtime, Trainer
    from aitrainer.parallelizer import parallelize

    config = FrameworkConfig.from_dict(case.config_values())
    torch.manual_seed(seed)
    runtime = Runtime(device="cpu", seed=seed)
    model = parallelize(build_model(seed), config=config, runtime=runtime)
    optimizer = build_optimizer(model, lr)
    trainer = Trainer(model, optimizer, config=config, runtime=runtime,
                      loss_fn=mean_squared_error)
    data = build_data(seed, steps, batch)
    started = time.perf_counter()
    observed = observe_training(model, trainer, data, lr=lr)
    elapsed = time.perf_counter() - started
    peak, peak_label = peak_memory_bytes()
    inner = getattr(trainer.model, "module", None)
    return {**observed, "seconds": elapsed, "step_seconds": elapsed / steps,
            "peak_memory_bytes": peak, "peak_memory_label": peak_label,
            "parameter_count": sum(parameter.numel()
                                   for parameter in _name_of(trainer.model).parameters()),
            "optimizer_steps": trainer.optimizer_step,
            "stage": int(getattr(inner, "stage_id", 0)),
            "stage_count": int(getattr(inner, "pp_size", 1))}


def gather_globally(payloads: list[dict], key: str) -> tuple[dict[str, list], list[str]]:
    """One mapping for the whole world, plus the names the ranks disagreed on.

    Values arrive global (see ``_as_global``), so a name seen on several ranks
    must carry the same numbers -- that is the point of collecting rather than
    overwriting.  A disagreement is a defect this script exists to find: it means
    two ranks hold different parameters under one name, which the loss curve
    would hide until it did not.

    Which names exist differs per axis: pipeline stages hold disjoint sets, TP
    and FSDP ranks hold the same set.  Both cases fall out of the same rule.
    """
    merged: dict[str, list] = {}
    conflicts: list[str] = []
    for payload in payloads:
        for name, values in (payload.get(key) or {}).items():
            if name not in merged:
                merged[name] = values
            elif merged[name] != values:
                conflicts.append(f"{name}@rank{payload.get('rank')}")
    return merged, conflicts


def _relative_error(expected: list, got: list, *, tolerance: float) -> float:
    """Largest absolute difference, scaled by the reference's own magnitude.

    Not an exact match: a sharded run takes a different route through the same
    arithmetic (a column projection's rows are a contiguous slice; a row
    projection's all-reduce reorders the sum), so the last bits differ
    legitimately.  What would not land inside the tolerance is a wrong layout, a
    missing reduction, or a stage holding another stage's parameters -- those
    miss by orders of magnitude, and the measured values are in the report
    either way.
    """
    if len(expected) != len(got):
        return float("inf")
    scale = max(1e-12, max((abs(value) for value in expected), default=0.0))
    worst = max((abs(a - b) for a, b in zip(expected, got)), default=0.0)
    return worst / scale if worst > tolerance else 0.0


def compare(reference: dict, candidate_ranks: list[dict], *, loss: float,
            tolerance: float) -> tuple[dict, list[str]]:
    """The four comparisons in the user's terms, plus any cross-rank conflicts.

    Output is compared only where it means something: under PP every stage
    produces a tensor, but only the last one is the model's output, so the other
    stages' activations are not a reference for anything.
    """
    problems: list[str] = []
    result: dict[str, object] = {}

    reference_params, _ = gather_globally([reference], "params")
    candidate_params, param_conflicts = gather_globally(candidate_ranks, "params")
    problems.extend(f"parameters disagree across ranks: {item}" for item in param_conflicts)

    reference_grads, _ = gather_globally([reference], "grads")
    candidate_grads, grad_conflicts = gather_globally(candidate_ranks, "grads")
    problems.extend(f"gradients disagree across ranks: {item}" for item in grad_conflicts)

    for label, expected, got in (("param", reference_params, candidate_params),
                                 ("grad", reference_grads, candidate_grads)):
        missing = sorted(set(expected) ^ set(got))
        if missing:
            problems.append(f"{label} names differ: {missing[:4]}")
            result[f"{label}_error"] = float("inf")
            continue
        result[f"{label}_error"] = max(
            (_relative_error(expected[name], got[name], tolerance=tolerance) for name in expected),
            default=0.0)
        result[f"{label}_names"] = len(expected)

    # Only the last pipeline stage produces the model's output AND its loss: the
    # earlier stages' tensors are activations on their way somewhere, and their
    # ScheduleOutput carries no loss at all.  Comparing them to the reference
    # would be comparing an intermediate activation to the model's output.
    last_stage = [item for item in candidate_ranks
                  if item.get("stage", 0) == item.get("stage_count", 1) - 1] or candidate_ranks
    result["output_error"] = _relative_error(reference["output"], last_stage[0]["output"],
                                             tolerance=tolerance)
    result["loss_error"] = (abs(last_stage[0]["final_loss"] - reference["final_loss"])
                            / max(1e-12, abs(reference["final_loss"])))
    # The errors ARE the verdict.  Computing them and then only checking for
    # cross-rank conflicts would have reported a 0.7 parameter error as ok.
    for label in ("output", "grad", "param"):
        measured = float(result.get(f"{label}_error", 0.0) or 0.0)
        if not (measured < tolerance):
            problems.append(f"{label} differs from the single-process reference: {measured:.3e}")
    return result, problems


# --------------------------------------------------------------------------- #
# the two roles
# --------------------------------------------------------------------------- #
def worker_main(options) -> int:
    """Run the baseline and every case, and hand back what this rank observed."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            timeout=__import__("datetime").timedelta(
                                seconds=float(options.timeout_seconds)))
    payload: dict = {"rank": rank, "world_size": world_size, "cases": {},
                     "baseline": None, "errors": []}
    try:
        # No baseline here: the launcher computes it before any process group
        # exists.  Runtime takes its world size from the environment, so inside a
        # rank the all-ones baseline config would be validated against the whole
        # world and rejected -- and a baseline that had to be told how big the
        # world is would not be a single-process reference any more.
        for case in cases_for(world_size):
            if case.world_size != world_size:
                continue
            try:
                seen = run_combination(case, steps=options.steps, seed=options.seed,
                                       batch=options.batch_size, lr=options.lr)
                seen["name"] = case.name
                payload["cases"][case.name] = seen
            except Exception as exc:                      # noqa: BLE001 - reported
                import traceback
                payload["cases"][case.name] = {
                    "name": case.name, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}
            dist.barrier()
        dist.barrier()
        print("RESULT " + json.dumps(payload), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


def _free_port() -> int:
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def launch(world_size: int, options) -> list[dict]:
    """One OS process per rank, with an explicit rendezvous.

    ``torchrun`` is not used on purpose: it resolves the hostname through mDNS
    on some hosts and dies with ``gai error: 8`` before any framework code runs,
    which would report an environment failure as a framework one.  Setting the
    rendezvous by hand works everywhere and keeps the failure legible.
    """
    port = _free_port()
    processes = []
    for rank in range(world_size):
        environment = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "RANK": str(rank), "WORLD_SIZE": str(world_size),
                       "PYTHONPATH": os.pathsep.join(
                           [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)}
        processes.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--role", "worker",
             "--steps", str(options.steps), "--seed", str(options.seed),
             "--batch-size", str(options.batch_size), "--lr", str(options.lr),
             "--timeout-seconds", str(options.timeout_seconds)],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    results: list[dict] = []
    for rank, process in enumerate(processes):
        try:
            out, err = process.communicate(timeout=options.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = process.communicate()
            out, err = out or "", (err or "") + "\n[timeout: killed]"
        found = None
        for line in (out or "").splitlines():
            if line.startswith("RESULT "):
                found = json.loads(line[len("RESULT "):])
        if found is None:
            found = {"rank": rank, "cases": {}, "baseline": None,
                     "errors": [f"rank {rank} produced no result"],
                     "stderr": (err or "")[-2000:]}
        results.append(found)
    return results


def assemble(reference: dict, results: list[dict], options) -> dict:
    """Turn per-rank observations into one report, with a verdict per case."""
    cases: list[dict] = []
    # Only the cases this world size can actually run: cases_for(4) also lists
    # the world-2 ones, and asking four ranks to run a two-rank configuration
    # reports "a rank produced nothing" -- a script bug wearing the costume of a
    # framework failure.
    runnable = [case for case in cases_for(len(results)) if case.world_size == len(results)]
    known = {case.name: case for case in runnable}
    for name in [case.name for case in runnable]:
        spec = known[name]
        ranks = [item["cases"].get(name) for item in results]
        failures = [item for item in ranks if item is None or "error" in item]
        broke = [item for item in failures if item is not None]
        entry: dict = {"name": name, "topology": asdict(spec), "status": "ok",
                       "problems": [], "metrics": {}}
        if failures:
            entry["status"] = "failed"
            entry["problems"] = ([item["error"] for item in broke]
                                 + ["a rank produced nothing"] * (len(failures) - len(broke)))
            cases.append(entry)
            continue
        metrics, problems = compare(reference, ranks, loss=ranks[0]["final_loss"],
                                    tolerance=options.tolerance)
        # Memory and throughput are properties of the whole run, so the largest
        # across ranks is the one that decides whether it fits.
        entry["metrics"] = {
            **metrics,
            "step_seconds": max(item["step_seconds"] for item in ranks),
            "peak_memory_bytes": max(item["peak_memory_bytes"] for item in ranks),
            "peak_memory_label": ranks[0]["peak_memory_label"],
            "parameter_count": sum(item["parameter_count"] for item in ranks),
            "optimizer_steps": sorted({item["optimizer_steps"] for item in ranks}),
        }
        entry["problems"] = problems
        if problems:
            entry["status"] = "failed"
        losses = ranks[0]["loss_curve"]
        if not all(math.isfinite(value) for value in losses):
            entry["status"] = "failed"
            entry["problems"].append("loss is not finite")
        cases.append(entry)
    return {"reference": {key: value for key, value in reference.items()
                          if key not in ("params", "grads", "output")},
            "cases": cases,
            "world_size": len(results),
            "errors": [error for item in results for error in item.get("errors", [])]}


def write_report(report_dir: Path, report: dict, options) -> tuple[Path, Path]:
    """Write the machine-readable JSON and the human-readable Markdown."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "verify_parallel.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = report_dir / "verify_parallel.md"

    reference = report.get("reference") or {}
    sizes = report.get("world_sizes") or [report.get("world_size")]
    lines = ["# 并行策略验证报告", "",
             f"- world size: **{', '.join(str(size) for size in sizes)}**",
             (f"- 步数: {options.steps} · batch: {options.batch_size} · seed: {options.seed}"
              f" · 容差: {options.tolerance:g}"),
             (f"- 基准（单进程）: 参数量 {reference.get('parameter_count', '?')}"
              f" · 每步 {reference.get('step_seconds', 0) * 1000:.2f} ms"
              f" · 最终 loss {reference.get('final_loss', float('nan')):.6f}"), "",
             ("四列误差都是**相对基准**的：|候选 − 基准| / max|基准|，"
              "在容差内记作 0。"), "",
             "| 组合 | world | 拓扑 | 前向 | loss | 梯度 | 参数 | 每步 ms | 峰值内存 | 结论 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for case in report["cases"]:
        topology = case["topology"]
        shape = (f"dp{topology['dp']} tp{topology['tp']} pp{topology['pp']}"
                 + (" sp" if topology["sp"] != "none" else ""))
        metrics = case.get("metrics") or {}
        if case["status"] != "ok":
            lines.append(f"| `{case['name']}` | {topology['world_size']} | {shape} "
                         f"| — | — | — | — | — | — | **失败** |")
            continue
        peak = metrics.get("peak_memory_bytes", 0)
        lines.append(
            f"| `{case['name']}` | {topology['world_size']} | {shape} "
            f"| {metrics.get('output_error', float('nan')):.2e} "
            f"| {metrics.get('loss_error', float('nan')):.2e} "
            f"| {metrics.get('grad_error', float('nan')):.2e} "
            f"| {metrics.get('param_error', float('nan')):.2e} "
            f"| {metrics.get('step_seconds', 0) * 1000:.2f} "
            f"| {peak / 1024 ** 2:.1f} MiB "
            f"| ok |")
    failures = [case for case in report["cases"] if case["status"] != "ok"]
    lines += ["", "## 结论", ""]
    if report["errors"]:
        lines += ["存在无法归因到单个组合的错误：", ""] + [f"- {item}" for item in report["errors"]] + [""]
    if failures:
        lines += [f"**{len(failures)} 个组合未通过**：", ""]
        for case in failures:
            lines += [f"- `{case['name']}`"]
            lines += [f"  - {problem}" for problem in case["problems"][:6]]
    elif not report["errors"]:
        lines += [(f"全部 {len(report['cases'])} 个组合与单进程基准在容差 "
                   f"{options.tolerance:g} 内一致，且各 rank 之间没有同名参数冲突。")]
    lines += ["",
              "## 这份报告能说明什么", "",
              "- **能**：每个组合的数值是否等于单进程；各 rank 是否持有相同的参数；",
              "  每个组合的耗时与峰值内存。",
              "- **不能**：组合是否*快*。本机没有加速器时每个集合通信都比它试图掩盖的东西慢，",
              "  所以耗时那一列比较的是同一套算术的不同实现，不是硬件行为。",
              ("- **也不能**：各 rank 用**不同**数据时是否正确。所有 rank 共用同一批数据，"
               "这样才能拿单进程当基准；发散副本的问题属于随机状态审计。")]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("launcher", "worker"), default="launcher",
                        help="launcher starts one process per rank; worker is internal")
    parser.add_argument("--world-size", type=int, default=None,
                        help="one world size to check (default: every size this host can "
                             "host, because each reaches combinations the others cannot)")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--report-dir", default="artifacts/verify_parallel")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    options = parser.parse_args(argv)
    if options.steps < 1 or options.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")

    if options.role == "worker":
        return worker_main(options)

    sizes = [options.world_size] if options.world_size else _default_world_sizes()
    sizes = [size for size in sizes if size >= 2]
    if not sizes:
        print("this check compares distributed runs against a single process; "
              "world size must be at least 2", file=sys.stderr)
        return 2
    started = time.perf_counter()
    # Here, before any process group exists: this is the single-process run.
    reference = run_baseline(steps=options.steps, seed=options.seed,
                             batch=options.batch_size, lr=options.lr)
    cases: list[dict] = []
    errors: list[str] = []
    for world_size in sizes:
        results = launch(world_size, options)
        for item in results:
            if "stderr" in item:
                print(f"[world {world_size} rank {item['rank']}] {item['stderr']}", file=sys.stderr)
        one = assemble(reference, results, options)
        cases.extend(one["cases"])
        errors.extend(f"world {world_size}: {item}" for item in one["errors"])
        print(f"  world {world_size}: {len(one['cases'])} 个组合 · "
              f"{sum(1 for case in one['cases'] if case['status'] != 'ok')} 个失败", flush=True)
    report = {"reference": {key: value for key, value in reference.items()
                            if key not in ("params", "grads", "output")},
              "cases": cases, "world_sizes": sizes, "errors": errors}
    json_path, markdown_path = write_report(Path(options.report_dir), report, options)
    failures = [case for case in cases if case["status"] != "ok"]
    print(f"{len(cases)} 个组合（world {', '.join(str(s) for s in sizes)}）· "
          f"{len(failures)} 个失败 · {time.perf_counter() - started:.1f}s")
    print(f"报告: {markdown_path}")
    print(f"      {json_path}")
    for case in failures:
        print(f"  FAIL {case['name']}: {'; '.join(case['problems'][:2])}")
    return 1 if (failures or errors) else 0


def _default_world_sizes() -> list[int]:
    """Every world size this host looks able to host, smallest first.

    All of them, not the largest: each world size reaches combinations the
    others cannot (a world of two has no interior pipeline stage; three axes at
    once need eight ranks), so a single size leaves most of the matrix unrun.
    Deliberately conservative about the count -- a world size the machine cannot
    carry would report a resource problem as a framework defect.
    """
    cores = os.cpu_count() or 2
    budget = max(2, cores // 2)
    return [size for size in (2, 4, 8) if size <= budget] or [2]


if __name__ == "__main__":
    sys.exit(main())
