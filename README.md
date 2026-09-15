# aiTrainer

aiTrainer is a small, model-agnostic PyTorch training foundation. Batch 0
provides immutable configuration, adapter protocols, a reproducible single
process trainer, portable checkpoints, and a manifest-based model-loading
contract. Later batches add explicit FSDP, TP, SP, PP, offload, checkpoint,
planner, and profiling interfaces with conservative capability boundaries.

```bash
pip install -e .
python examples/tiny_transformer/train.py
pytest
```

Batch 1 adds the explicit runtime/topology and FSDP wrapper. A real
multi-rank run must opt in to FSDP and use matching mesh dimensions:

```bash
torchrun --standalone --nproc_per_node=2 examples/fsdp_train/train.py
```

This has been executed: `examples/fsdp_train/train.py` completes at two ranks
over Gloo on CPU (`completed 4 optimizer steps`, both ranks exit 0). Where
`torchrun` cannot start -- it resolves the local hostname through mDNS on macOS
and dies with `gai error: 8`, which is a limitation of `torchrun` and not of
multi-rank -- set the rendezvous variables explicitly instead:

```bash
for r in 0 1; do
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=$r \
    python examples/fsdp_train/train.py &
done; wait
```

`tests/multirank/` does exactly this to run one OS process per rank, and
`tests/unit/test_multirank_gloo.py` uses it to check the FSDP, TP, SP and
process-group paths against real multi-process jobs.

Batch 2 adds explicit tensor layouts, synchronous autograd collectives and
Megatron-style Column/Row Parallel Linear layers. TP/FSDP uses orthogonal TP
and DP groups; the composition entry applies TP sharding before FSDP wrapping.

```bash
python examples/tp_train/train.py
```

Batch 3 adds fixed-shape and variable-length sequence-parallel utilities,
Ulysses attention exchange, RoPE offsets, and sequence-parallel LayerNorm.
SP must be paired with an explicit TP size and process group. The
variable-length path (`seq_lens`) is available at `world_size == 1` only: at
`world >= 2` it raises `SPConfigurationError` rather than returning attention
computed from a mask that does not model where each rank's padding lands after
the head/sequence exchange (it previously returned wrong values there -- max abs
error 0.42 against dense SDPA). Pad the global sequence before sharding, or pass
`seq_lens=None` for dense, evenly-divisible batches.

Ulysses also requires NCCL: `all_to_all` has no Gloo implementation, so at
`world_size > 1` every Ulysses path raises `RuntimeError: Backend gloo does not
support alltoall` on a CPU/Gloo job no matter how it is configured. The
sequence-parallel paths that use TP collectives instead (scatter/gather,
`SequenceParallelLayerNorm`) do run on Gloo.

Batch 4 adds ordered pipeline stage planning, tensor metadata, synchronous P2P,
GPipe, and non-interleaved 1F1B schedules. Virtual/interleaved stages remain
outside the stable boundary; Batch 9 adds opt-in preposted P2P overlap with a
formal synchronous fallback.

Batch 5 adds atomic checkpoint directories with `READY` and checksum markers,
RNG/optimizer/scheduler/scaler/sampler restoration, explicit world-size
compatibility checks, dense-to-sharded conversion, data contracts, structured
diagnostics, and benchmark artifacts.

Batch 8 adds the opt-in `Trainer.from_model` entry point, Llama/GPT/BERT-shaped
reference adapters (randomly initialised -- no pretrained weights are loaded),
deterministic topology/stage planning through `suggest_plan`, and read-only CLI
commands:

```bash
aitrainer inspect-topology
aitrainer dry-run  --config examples/config.json
aitrainer validate --config examples/config.json --world-size 1
```

`suggest_plan` is a library entry point; the CLI's `dry-run` reports the config
and the capability matrix, not a plan. `examples/config.json` is a minimal
configuration that validates as shipped.

Automatic planning is disabled by default. A report never rewrites an explicit
parallel configuration; applying a reviewed candidate requires
`PlanningConfig(allow_rewrite=True)`.

Batch 9 exposes independent advanced-overlap switches through
`AdvancedOverlapConfig`: TP bulk collectives, backward-ready gradient buckets,
trace-validated parameter prefetch, PP P2P, and bounded H2D/D2H transfer. Every
switch is disabled by default, FSDP reducer conflicts are rejected at startup,
and all handles are drained before checkpoint/exit. `Profiler` exposes timeline,
wait/overlap, pipeline-bubble, GPU-idle, allocation and buffer-pool methods, but
nothing in the trainer instantiates it yet, so you have to create and populate
one yourself.

Note that the advanced-overlap switches currently have **no effect at all** --
not even on bookkeeping: for switches whose required axis matches the configured
topology, constructing a `Trainer` with one enabled yields byte-identical
`overlap.summary()` / `offload.stats()` to the all-off baseline, and
`enable_microbatch_interleave` plus `enable_optimizer_param_gather_overlap` are
not wired into the controller in any form. (Switches whose axis is absent --
`enable_tp_bulk_overlap`, `enable_pp_p2p_overlap`, `enable_microbatch_interleave`
at the default single-rank topology -- abort construction with
`UnsupportedCombinationError` instead.) Treat them as reserved configuration, not
as knobs. The experimental
`MicrobatchInterleaveScheduler` handles exactly two static TP microbatches and
rejects PP, dynamic-control-flow and activation-offload combinations at startup,
but it is not reachable from `Trainer`.

For a complete server-side validation matrix, copy the repository and run:

```bash
bash scripts/run_all_tests.sh --world-size 2 --report-dir artifacts/server_report
```

Use `--world-size 4` (or `8`) to execute the PP combination entry as a real
torchrun job; a two-rank run records that entry as structurally validated but
blocked by its required `tp=2, pp=2` topology.

The runner executes static checks, unit tests, distributed boundary tests,
`torchrun` tests, and performance baselines when their dependencies exist. It
also runs `scripts/overlap_smoke.py` by default, covering the Batch 9
AsyncOp state machine, drain, gradient bucket, ping-pong generation,
trace-driven parameter lifecycle and the two-microbatch experimental contract.
Use `--skip-overlap-smoke` only when this local contract check is intentionally
excluded. It
also runs `scripts/parallel_matrix.py`, which covers the requested
`FSDP`, `FSDP+SP`, `FSDP+SP+TP`, and `FSDP+SP+TP+PP` entries. Pure FSDP is
measured with real multi-rank forward/backward, parameter-update correctness,
global throughput, and per-rank CUDA allocator peaks. The combined entries
exercise the PP → TP/SP → FSDP composition path and retain concrete runtime
status and failure reasons. It also records
an asynchronous all-reduce plus independent matrix-compute probe, including
communication time, compute time, estimated overlap seconds, and overlap
ratio. This probe is instrumentation for the communication backend and is
clearly labelled separately from native FSDP internals.

It writes `results.json`, `results.csv`, per-command logs, a
`training_benchmark.json` containing forward/backward correctness, steps/s,
samples/s, CPU RSS and CUDA peak allocated/reserved memory, and a
`report.html` with charts and a parallel-combination matrix. `report.html`
links each command's stdout/stderr as relative hrefs into the generated
`logs/` directory, so keep those two together if you move the report.
The distributed phase also runs
`scripts/distributed_correctness.py` under `torchrun` to compare multi-rank
forward values, gradients and parameter updates against a single-rank
reference. The training benchmark uses three fixed-seed repeats by default
(`--benchmark-repeats` changes this). Missing `torch`, `pytest` or CUDA is
reported as blocked/skipped and causes a non-zero exit rather than being treated
as a successful validation. A missing `torchrun` is handled differently: the
runner falls back to `python -m torch.distributed.run` and the distributed
entry is attempted rather than recorded as blocked.

Run it with the interpreter that has PyTorch:
`PYTHON=/path/to/venv/bin/python bash scripts/run_all_tests.sh --help` lists the
flags, including `--skip-distributed`, `--skip-performance`, `--skip-benchmark`,
`--skip-parallel-matrix`, `--benchmark-steps`, `--benchmark-batch-size`,
`--parallel-matrix-steps`, `--pytest-args` and `--strict`. Note that
`--skip-distributed` and `--skip-parallel-matrix` skip the headline matrix
claims entirely.
