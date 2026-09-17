# aiTrainer

A small, model-agnostic PyTorch training foundation: immutable configuration,
adapter protocols, a reproducible trainer, portable checkpoints, and explicit
FSDP / TP / SP / PP / offload interfaces with conservative capability boundaries.

**Architecture**: [`docs/架构.md`](docs/架构.md) has the layered module map and
the dependency contracts that CI enforces.
[`docs/GPU验证清单.md`](docs/GPU验证清单.md) lists what cannot be settled on a
CPU/Gloo host and how to check it on a real one.
`docs/路线图.md` and `docs/开发方案与开发规划.md` describe a *target* state, not
the code as it stands; `docs/代码审计报告.md` records the defect history and what
was verified empirically.

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
| `capability.validate_combinations` | rules that span siblings (`dp_size>1` needs FSDP, `compile.enabled` is refused, an overlap switch needs its axis) |
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

**TP** — `torch.distributed.tensor.parallel.parallelize_module` with
`ColwiseParallel` / `RowwiseParallel` styles, which projection is which coming
from `TransformerTPPlan`, over an explicit TP group. The parameters become
`DTensor`s, and that is load-bearing rather than incidental: it is what lets a
sharded checkpoint tell one logical tensor apart from one rank's slice of it.
`parallelize()` applies TP before FSDP wrapping, using orthogonal groups, and a
plan that names no module is **refused** rather than leaving every rank a full
replica.

**SP** — sequence-parallel activation of each norm over the TP group
(`sequence_parallel_styles`). Norms are matched **structurally**, by type, so
sequence parallelism imposes no naming convention on the model; what it covers
is `LayerNorm` and `RMSNorm`. The style is indifferent to which of them it
wraps — it shards the *sequence* axis and leaves the norm's own axis, the last
one, whole on every rank — so anything normalising over the hidden dimension
applies, and adding a type is a one-line change to `sequence_parallel_types`.
A model whose norms are none of these is **refused**, not run with SP quietly
absent: that was the old behaviour, and it is invisible in the loss because SP
is an activation-layout optimisation. The sharding is written as a `Replicate ->
Shard(1)` layout transition so DTensor's autograd supplies the gather and its
reduce-scatter; a hand-rolled local `chunk` is measurably wrong, because its
backward hands each rank only its own slice's contribution to the input
gradient. Dropout stays **outside** the region on purpose — inside it, every
rank draws the first block's mask. `sp_backend` is an on/off switch, not a
choice of implementation: `megatron` and `ulysses` used to run identical code,
and `ulysses` is now refused at validation instead of being silently
reinterpreted.

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
  support alltoall` on a CPU/Gloo job however it is configured. That is why
  `sp_backend='ulysses'` is refused rather than accepted: the head exchange is a
  separate explicit API (`UlyssesAttention`, `distributed_attention`), and the
  config value cannot select it on a backend where it cannot run. The norm
  sequence sharding above uses only DTensor layout transitions and does run on
  Gloo.

**PP** — ordered stage planning, explicit tensor metadata, synchronous P2P, and
`GPipeSchedule` / `OneFOneBSchedule` (non-interleaved). `parallel/pp_schedule.py`
is the only schedule implementation; `PipelineStage` delegates to it. Virtual and
interleaved stages are outside the stable boundary.

The split divides the model's **direct children**, so a model whose `forward` is
not a chain of them needs to say what the units are. A HuggingFace model is the
usual case: `LlamaForCausalLM`'s children are the trunk and the head, and its
layers are one level further down, so the top-level split would put the whole
transformer on stage 0. `execution_order` is the answer, and
`parallel.pp_shapes.execution_units` documents the shape it returns — the ordered
units including the glue, named without dots:

```python
def llama_execution_order(model):
    return [("embed_tokens", model.model.embed_tokens),
            *[(f"layer_{index}", layer)
              for index, layer in enumerate(model.model.layers)],
            ("norm", model.model.norm), ("lm_head", model.lm_head)]

stages = parallelize(model, config=config, runtime=runtime,
                     execution_order=llama_execution_order)
```

Without one, a split that would hand a stage a child **which cannot run at all**
— a bare `nn.Module`, or the `nn.ModuleList` every HuggingFace model holds its
layers in — is refused. That check is a proof rather than a heuristic: no
assignment of those children to stages can execute.

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
and buffer-pool methods, and nothing instantiates it for you — create one and
wrap the block you want measured:

```python
profiler = Profiler()
with profiler.capture():
    trainer.fit(batches, epochs=1)
profiler.summary()["overlap_ratio"]     # measured, not structural
```

`capture()` reads a `torch.profiler` trace back, because the collectives worth
measuring are the ones inside DTensor and FSDP2 — autograd nodes with no call
site to hang a timer on. Matching is on the transfer itself: one collective
appears in the trace as a nest (`FSDP::all_gather` wrapping `c10d::_allgather_base_`,
two `*_copy_in/out`, and the `gloo:`/`nccl:` transfer), so counting by operation
name would report it three times over. Exposed time is the part of each transfer
no compute covers. On CPU/Gloo the backend transfers on a worker thread, so the
split is real there; on CUDA it is the number the overlap work exists to move.

`kernels/fused.py` holds numerically transparent reference implementations of
norm/MLP/residual/AdamW.  They are a reference, not a dispatch layer: a
`KernelBackend` registry once sat on top of them, and it was deleted because
PyTorch's own dispatcher already picks the fast implementation for these
operators (`F.layer_norm` is the CUDA fused kernel, SDPA dispatches to
flash/mem-efficient/math), no third-party backend was ever registered, and
nothing read the record of which one ran.

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
# Correctness against a single-process baseline, plus throughput and peak
# memory, for every combination this host can run -- and a report.
python scripts/verify_parallel.py --report-dir artifacts/verify

bash scripts/run_all_tests.sh --world-size 2 --report-dir artifacts/server_report
```

`verify_parallel.py` trains the same small model every way this host can run and
compares each against a single-process baseline on forward output, loss, the
parameter update and the final weights.  Every combination also saves a sharded
checkpoint, loads it into a second trainer, and requires the parameters back bit
for bit and the RESUMED step to land where the uninterrupted one did -- which is
what covers the optimizer's momentum, the step counters and every RNG stream.  A
report ends with a table of every combination against every other, so "these are
equivalent ways to run one training" is stated without reference to a baseline.

It has found four real defects that every existing test missed, plus the one
that showed up in a model-decoupling review: sequence parallelism over a model
whose norm it does not know used to run with SP configured and silently not
applied, which no loss curve can show.  The fourth is
the one worth reading the audit report for: with `tp_size>1`, sequence
parallelism on and `dp_size=1`, a sharded checkpoint restored the parameters
bit for bit and the optimizer momentum wrongly, so a resumed run continued 2e-01
away from the run that wrote it.  The cause was not the checkpoint.  Sequence
parallelism feeds each norm a *slice* of the sequence, so the norm's replicated
parameters accumulate a `Partial` (per-rank partial sum) gradient that autograd
never reduces -- the parameter update comes out right anyway, because DTensor
reduces when it computes the new parameter value, so only the optimizer's state,
gradient clipping, and anything checkpointing them were wrong.  FSDP hid it by
wrapping the parameters again, and `tp_size=1` never produces it.  A first fix
using a `post_accumulate_grad_hook` worked in isolation and was dead in the
framework's own path: `Trainer.__init__` calls `model.to()`, one call of which
stops that hook firing on a DTensor parameter, silently, forever.

The other three.  Both pipeline
schedules backpropagated a stage's received *activation* instead of its produced
*output*, leaving every interior stage's parameters at `grad is None` for the
whole run while the loss stayed correct and `backward_complete` reported true (at
pp=2 there is no interior stage, so no two-stage run could see it).  And
`load_sharded` on a pipeline returned successfully while dropping four scalar
keys from the optimizer's parameter groups, so the failure surfaced only on the
NEXT step, as `KeyError: 'momentum'` from inside `torch.optim`.

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
