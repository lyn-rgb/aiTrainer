# aiTrainer

A small, model-agnostic PyTorch training foundation: immutable configuration,
adapter protocols, a reproducible trainer, portable checkpoints, and explicit
FSDP / TP / SP / PP / offload interfaces with conservative capability boundaries.

**Architecture**: [`docs/架构.md`](docs/架构.md) has the layered module map and
the dependency contracts that CI enforces. `docs/路线图.md` and
`docs/开发方案与开发规划.md` describe a *target* state, not the code as it stands;
`docs/代码审计报告.md` records the defect history and what was verified empirically.

```bash
pip install -e .
python examples/tiny_transformer/train.py
pytest
```

---

## Configuration and validation

`FrameworkConfig` is an immutable tree: `parallel`, `precision`, `compile`,
`fsdp`, `offload`, `overlap`, `planning`, plus scalar fields. Construct it in
Python or parse JSON with `FrameworkConfig.from_dict` — the nested dataclasses
are discovered from the type annotations, so no field is special-cased.

Validation happens in three places, each with one job:

| Layer | Owns |
|---|---|
| `config/schema.py` | per-config invariants (types, ranges, self-consistency) |
| `capability.validate_combinations` | rules that span siblings (`dp_size>1` needs FSDP, microbatch interleave rejects PP, ...) |
| `capability.validate_capabilities` | the public entry point: schema first, then combinations |

`trainer` constructs once and validates once. `capability_matrix()` reports what
is stable, experimental, or unsupported, each entry with its honest reason.

`ConfigPreset` provides `single_gpu()`, `fsdp()`, `tp()`, `tp_fsdp()`, `pp()`,
`memory_saving()` and `performance()`.

## Runtime and topology

`Runtime` owns the process group, device selection, seeding and cleanup, and is
a context manager. On CPU or a single process it creates no rendezvous. The seed
is applied when the runtime is created, so **build the model after it** if you
want reproducible initialisation — `Trainer.from_model` does this for you;
`Trainer(model, ...)` does not, because the model you pass in was already built.

`RankMapping` describes an explicit logical `(pp, dp, tp)` mesh; `DeviceMeshManager`
turns it into the orthogonal process groups. `inspect-topology` prints what was
detected.

## Parallelism

**TP** — Megatron-style `ColumnParallelLinear` / `RowParallelLinear` over an
explicit TP group, with synchronous autograd-aware collectives. `parallelize()`
applies TP sharding before FSDP wrapping, using orthogonal groups.

**SP** — fixed-shape and variable-length sequence-parallel utilities, Ulysses
head exchange, RoPE offsets, `SequenceParallelLayerNorm`.

Two constraints worth knowing before you hit them:

- The **variable-length** (`seq_lens`) Ulysses path is available at
  `world_size == 1` only. At `world >= 2` it raises `SPConfigurationError`
  instead of returning attention computed from a mask that does not model where
  each rank's padding lands after the head/sequence exchange — it previously
  returned wrong values there (max abs error 0.42 against dense SDPA). Pad the
  global sequence before sharding, or pass `seq_lens=None` for dense,
  evenly-divisible batches.
- **Ulysses requires NCCL.** `all_to_all` has no Gloo implementation, so at
  `world_size > 1` every Ulysses path raises `RuntimeError: Backend gloo does not
  support alltoall` on a CPU/Gloo job however it is configured. The
  sequence-parallel paths that use TP collectives instead (scatter/gather,
  `SequenceParallelLayerNorm`) do run on Gloo.

**PP** — ordered stage planning, explicit tensor metadata, synchronous P2P, and
`GPipeSchedule` / `OneFOneBSchedule` (non-interleaved). `parallel/pp_schedule.py`
is the only schedule implementation; `PipelineStage` delegates to it. Virtual and
interleaved stages are outside the stable boundary.

**FSDP** — `FULL_SHARD` over the explicit DP group, with TP/SP/PP axes kept
orthogonal. A multi-rank run must opt in and use matching mesh dimensions:

```bash
torchrun --standalone --nproc_per_node=2 examples/fsdp_train/train.py
```

That has been executed: it completes at two ranks over Gloo on CPU (`completed 4
optimizer steps`, both ranks exit 0). Where `torchrun` cannot start — it resolves
the local hostname through mDNS on macOS and dies with `gai error: 8`, a
limitation of `torchrun` and not of multi-rank — set the rendezvous explicitly:

```bash
for r in 0 1; do
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=$r \
    python examples/fsdp_train/train.py &
done; wait
```

`tests/multirank/` does exactly this to run one OS process per rank;
`tests/unit/test_multirank_gloo.py` uses it to check the process-group, TP, SP,
FSDP and pipeline paths against real multi-process jobs.

## Checkpoints

Two formats, both of them complete resumes. `save_checkpoint` writes one atomic
directory per rank (a `READY` marker and per-file checksums); `save_sharded` is
the opt-in `torch.distributed.checkpoint` path with one collective call and one
shared path. Both carry the model, optimizer, scheduler, scaler, sampler
position, step counters and both RNG streams, and both have explicit world-size
compatibility checks. `CheckpointConverter` handles dense-to-sharded conversion
and `ModelLoader` the manifest-based loading contract.

A sharded load also refuses a checkpoint that holds a namespace the restoring
state never mentions (a scheduler, say), rather than reading everything else and
dropping it without saying so.

## Offload and overlap

CPU offload is explicit and opt-in through `OffloadConfig` (optimizer state,
parameters, activations). `OffloadManager` is the composition root; `Trainer`
calls `fetch`/`release` around each step.

`AdvancedOverlapConfig` exposes independent overlap switches — but they are
**rejection-only as of this version**. Turning one on does not turn the feature
on; it only makes startup reject combinations that genuinely cannot work (for
example gradient-bucket overlap together with FSDP's own reducer). They are kept
deliberately: removing them would silently widen the set of accepted
configurations. Treat them as reserved configuration, not as knobs.

`Profiler` exposes timeline, wait/overlap, pipeline-bubble, GPU-idle, allocation
and buffer-pool methods, but nothing instantiates it yet — create and populate
one yourself. `MicrobatchInterleaveScheduler` handles exactly two static TP
microbatches and rejects PP, dynamic-control-flow and activation-offload
combinations at startup, but it is not reachable from `Trainer`.

## Training

`Trainer` needs an optimizer and iterates batches:

```python
from aitrainer import ConfigPreset, Trainer

trainer = Trainer(model, optimizer, config=ConfigPreset.single_gpu())
trainer.fit(batches)
trainer.save_checkpoint("run/checkpoint")
```

`Trainer.from_model` is the higher-level entry: it seeds before building, accepts
an adapter or a module, can create the optimizer from a factory or class, and
applies automatic parallel wrapping when the config asks for it. Llama/GPT/BERT-
shaped reference adapters are available and are **randomly initialised** — no
pretrained weights are loaded.

A trailing partial accumulation window is **dropped**, not flushed: its
microbatches were accumulated under `no_sync()`, so with `dp_size>1` their
gradients were never reduced, and stepping on them both diverged the replicas and
mis-scaled the update.

## CLI and planning

```bash
aitrainer inspect-topology
aitrainer dry-run  --config examples/config.json
aitrainer validate --config examples/config.json --world-size 1
```

`suggest_plan` is a library entry point; the CLI's `dry-run` reports the config
and the capability matrix, **not** a plan. `examples/config.json` is a minimal
configuration that validates as shipped. Automatic planning is disabled by
default, a report never rewrites an explicit parallel configuration, and applying
a reviewed candidate requires `PlanningConfig(allow_rewrite=True)`.

## Validation matrix

```bash
bash scripts/run_all_tests.sh --world-size 2 --report-dir artifacts/server_report
```

Use `--world-size 4` (or `8`) to execute the PP combination entry as a real job;
a two-rank run records it as structurally validated but blocked by its required
`tp=2, pp=2` topology.

The runner executes static checks, unit tests, distributed boundary tests,
`torchrun` tests and performance baselines when their dependencies exist. It also
runs `scripts/overlap_smoke.py` (the `AsyncOp` state machine, drain, gradient
bucket, ping-pong generation, trace-driven parameter lifecycle, and the
two-microbatch experimental contract — skip with `--skip-overlap-smoke` only when
that local contract check is intentionally excluded) and `scripts/parallel_matrix.py`,
which covers the `FSDP`, `FSDP+SP`, `FSDP+SP+TP` and `FSDP+SP+TP+PP` entries.
Pure FSDP is measured with real multi-rank forward/backward, parameter-update
correctness, global throughput and per-rank CUDA allocator peaks; the combined
entries exercise the PP → TP/SP → FSDP composition path. An asynchronous
all-reduce plus independent matrix-compute probe records communication time,
compute time, estimated overlap seconds and overlap ratio — instrumentation for
the communication backend, labelled separately from native FSDP internals.

It writes `results.json`, `results.csv`, per-command logs, a
`training_benchmark.json` (forward/backward correctness, steps/s, samples/s, CPU
RSS, CUDA peak allocated/reserved) and a `report.html` with charts and the
parallel-combination matrix. `report.html` links each command's stdout/stderr as
relative hrefs into the generated `logs/` directory, so keep those two together
if you move the report. The distributed phase also runs
`scripts/distributed_correctness.py` under `torchrun`, comparing multi-rank
forward values, gradients and parameter updates against a single-rank reference.
The benchmark uses three fixed-seed repeats by default (`--benchmark-repeats`).
Missing `torch`, `pytest` or CUDA is reported as blocked/skipped and causes a
non-zero exit. A missing `torchrun` is different: the runner falls back to
`python -m torch.distributed.run` and attempts the entry rather than recording it
as blocked.

Run it with the interpreter that has PyTorch:
`PYTHON=/path/to/venv/bin/python bash scripts/run_all_tests.sh --help` lists the
flags, including `--skip-distributed`, `--skip-performance`, `--skip-benchmark`,
`--skip-parallel-matrix`, `--benchmark-steps`, `--benchmark-batch-size`,
`--parallel-matrix-steps`, `--pytest-args` and `--strict`. Note that
`--skip-distributed` and `--skip-parallel-matrix` skip the headline matrix claims
entirely.
