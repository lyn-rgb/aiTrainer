"""Regression tests for the defects recorded in docs/代码审计报告.md.

Each test pins a bug that was fully reproducible before the fix.  Tests whose
defect already had coverage (collectives arity, the tuple-batch contract, the
overlap baseline) live in their original files.
"""

import pytest

torch = pytest.importorskip("torch")

from aitrainer.config import (
    FrameworkConfig,
    FSDPConfig,
    MixedPrecisionConfig,
    OffloadConfig,
    ParallelConfig,
    PlanningConfig,
)
from aitrainer.data import model_inputs
from aitrainer.diagnostics import dry_run
from aitrainer.memory import PinnedBufferPool
from aitrainer.offload.parameter import ParameterOffloader
from aitrainer.overlap.gradient import GradientBucketReducer
from aitrainer.planner import PlanCandidate, apply_candidate, suggest_plan


class _Layer:
    def named_parameters(self): return iter(())
    def parameters(self): return iter(())


class _StackedModel:
    """Minimal ordered model: suggest_plan needs children() with >= 1 layer."""

    def __init__(self, layers: int = 4) -> None:
        self._layers = [_Layer() for _ in range(layers)]

    def children(self): return iter(self._layers)
    def named_children(self): return iter([(f"l{i}", l) for i, l in enumerate(self._layers)])
    def named_parameters(self): return iter(())
    def parameters(self): return iter(())


# --- supervision never reaches forward ---------------------------------------

def test_model_inputs_strips_supervision_and_keeps_inputs():
    inputs, labels = torch.zeros(2, 3), torch.zeros(2)
    assert model_inputs((inputs, labels)) == ((inputs,), {})
    assert model_inputs({"input_ids": inputs, "labels": labels}) == ((), {"input_ids": inputs})
    # A Mapping that is only supervision is passed through rather than emptied.
    assert model_inputs({"labels": labels}) == ((), {"labels": labels})
    assert model_inputs(inputs) == ((inputs,), {})


# --- parameter offload (was KeyError on the first train_step) ---------------

def test_parameter_offload_runs_a_step_and_updates_weights():
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
    config = FrameworkConfig(device="cpu",
                             offload=OffloadConfig(enabled=True, parameter=True, max_cpu_bytes=1 << 20))
    from aitrainer import Trainer
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3), config=config,
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    before = [p.detach().clone() for p in model.parameters()]
    trainer.train_step((torch.randn(8, 8), torch.randn(8, 4)))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    trainer.close()


def test_same_shaped_parameters_keep_distinct_records_and_leases():
    """BufferKey collisions used to collapse records while leaking their buffers."""
    module = torch.nn.Sequential(*[torch.nn.Linear(8, 8) for _ in range(3)])
    pool = PinnedBufferPool(1 << 20, pin_memory=False)
    offloader = ParameterOffloader(pool=pool)
    expected = sum(1 for _ in module.parameters())
    assert offloader.register_module(module) == expected
    assert len(offloader._records) == expected
    assert pool.stats()["in_flight"] == expected          # no orphaned leases
    offloader.release_all()
    offloader.clear()                                      # must not raise


# --- planner ----------------------------------------------------------------

def test_suggest_plan_returns_candidates_when_enabled():
    """`any(<bool>)` raised TypeError, so an enabled planner never returned."""
    config = FrameworkConfig(planning=PlanningConfig(enabled=True))
    for world in (1, 4):
        report = suggest_plan(_StackedModel(4), config=config, world_size=world)
        assert report.enabled and report.candidates and report.selected is not None


def test_suggest_plan_reports_a_stage_plan_for_a_pipeline_candidate():
    report = suggest_plan(_StackedModel(4),
                          config=FrameworkConfig(parallel=ParallelConfig(pp_size=2, pp_schedule="gpipe"),
                                                 planning=PlanningConfig(enabled=True)),
                          world_size=2)
    assert report.enabled and report.selected.pp_size == 2


def test_apply_candidate_preserves_fields_it_does_not_name():
    """Rebuilding the dataclasses dropped sp_backend, pp_schedule and FSDP settings."""
    config = FrameworkConfig(
        planning=PlanningConfig(enabled=True, allow_rewrite=True),
        parallel=ParallelConfig(dp_size=2, tp_size=2, pp_size=1, sp_backend="ulysses"),
        fsdp=FSDPConfig(enabled=True, forward_prefetch=True, execution_trace_complete=True,
                        mixed_precision=MixedPrecisionConfig(enabled=True, dtype="float16")))
    updated = apply_candidate(config, PlanCandidate(4, 2, 1, "fsdp", 0.0, ("t",)))
    assert updated.parallel.sp_backend == "ulysses"
    assert updated.fsdp.forward_prefetch is True
    assert updated.fsdp.execution_trace_complete is True
    assert updated.fsdp.mixed_precision.enabled is True
    assert updated.fsdp.mixed_precision.dtype == "float16"
    assert updated.parallel.dp_size == 4
    updated.validate(world_size=8)
    # the input config must not be mutated
    assert config.parallel.dp_size == 2 and config.fsdp.mixed_precision.dtype == "float16"


# --- gradient bucket reducer (reduced once per process lifetime) ------------

class _FakeTensor:
    def numel(self): return 8
    def element_size(self): return 4


def test_gradient_bucket_reducer_reduces_every_window():
    calls: list[int] = []
    reducer = GradientBucketReducer(1024, accumulation_steps=1,
                                    reduce_fn=lambda values: calls.append(len(values)))
    for _ in range(5):
        reducer.register("w", _FakeTensor())
        reducer.mark_microbatch_end()
        reducer.finish_grad_sync()
    assert len(calls) == 5
    assert sum(len(bucket.tensors) for bucket in reducer.buckets) == 0


def test_gradient_bucket_reducer_honours_accumulation_steps():
    calls: list[int] = []
    reducer = GradientBucketReducer(1024, accumulation_steps=4,
                                    reduce_fn=lambda values: calls.append(len(values)))
    for _ in range(12):
        reducer.register("w", _FakeTensor())
        reducer.mark_microbatch_end()
        reducer.finish_grad_sync()
    assert len(calls) == 3


def test_gradient_bucket_reducer_refuses_to_flush_without_a_reducer():
    """Flushing with reduce_fn=None cleared buckets and dropped gradients silently."""
    reducer = GradientBucketReducer(1024, accumulation_steps=1)
    reducer.register("w", _FakeTensor())
    with pytest.raises(ValueError, match="reduce_fn"):
        reducer.finish_grad_sync()


# --- reproducibility and diagnostics ----------------------------------------

def test_from_model_seeds_before_building_so_weights_repeat():
    from aitrainer import Trainer
    from aitrainer.plugins.models import GPTAdapter, TransformerModelConfig

    def weights(seed: int) -> list[torch.Tensor]:
        adapter = GPTAdapter(TransformerModelConfig(vocab_size=64, hidden_size=16, heads=2, layers=1))
        trainer = Trainer.from_model(adapter, config=FrameworkConfig(seed=seed, device="cpu"))
        result = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.close()
        return result

    same = weights(123)
    assert all(torch.equal(a, b) for a, b in zip(same, weights(123)))
    assert any(not torch.equal(a, b) for a, b in zip(same, weights(999)))


def test_dry_run_validates_a_multi_rank_config():
    """dry_run had no world_size, so every distributed config reported a false mismatch."""
    config = FrameworkConfig(parallel=ParallelConfig(dp_size=8), fsdp=FSDPConfig(enabled=True))
    result = dry_run(config)
    assert result["status"] == "ok" and result["world_size"] == 8
    assert dry_run(config, world_size=8)["status"] == "ok"


# --- process-group creation order (>=3 ranks deadlocked for 30 min) ---------

@pytest.mark.parametrize("dp,tp,pp", [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 2, 1), (2, 2, 2), (4, 2, 2)])
def test_group_creation_plan_is_canonical_for_every_rank(dp, tp, pp):
    """All ranks must issue identical new_group calls, in the same order.

    Creating only each rank's own groups gave per-process call counters that
    collided in the store: one index carried groups of different sizes, no rank
    observed the expected barrier count, and every participant blocked for the
    full 30-minute gloo timeout from 3-4 ranks up.
    """
    from aitrainer.parallel.groups import group_creation_plan
    from aitrainer.topology import make_rank_mapping

    world = dp * tp * pp
    mapping = make_rank_mapping(world, pp_size=pp, dp_size=dp, tp_size=tp)
    plan = group_creation_plan(mapping)
    assert group_creation_plan(mapping) == plan                    # deterministic
    assert len(plan) == (world // pp) + (world // dp) + (world // tp)
    for axis, size in (("pp", pp), ("dp", dp), ("tp", tp)):
        groups = [ranks for name, ranks in plan if name == axis]
        assert len(groups) == world // size
        assert all(len(set(ranks)) == size for ranks in groups)
        for rank in range(world):
            assert sum(rank in ranks for ranks in groups) == 1     # exactly one group per axis


def test_group_creation_plan_matches_the_verified_canonical_sequence():
    """Pin the exact sequence a 4-rank dp=2/tp=2 run must issue (4/4 passed locally)."""
    from aitrainer.parallel.groups import group_creation_plan
    from aitrainer.topology import make_rank_mapping

    plan = group_creation_plan(make_rank_mapping(4, pp_size=1, dp_size=2, tp_size=2))
    assert plan == (
        ("pp", (0,)), ("pp", (1,)), ("pp", (2,)), ("pp", (3,)),
        ("dp", (0, 2)), ("dp", (1, 3)),
        ("tp", (0, 1)), ("tp", (2, 3)),
    )


# --- accumulation boundary: clipping and GradScaler skips -------------------

def test_grad_scaler_skip_does_not_advance_schedule_or_optimizer_step():
    """A skipped update must not advance the LR schedule or optimizer_step.

    GradScaler is only auto-enabled on CUDA, so this installs an enabled CPU
    scaler to exercise the skip path; an inf loss makes it skip.
    """
    from aitrainer import Trainer

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(model, optimizer, config=FrameworkConfig(device="cpu"), scheduler=scheduler,
                      loss_fn=lambda output, batch: output.sum() * float("inf"))
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)

    before = [p.detach().clone() for p in model.parameters()]
    before_lr = optimizer.param_groups[0]["lr"]
    result = trainer.train_step((torch.randn(2, 4),))

    assert trainer.optimizer_step == 0
    assert optimizer.param_groups[0]["lr"] == before_lr
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    # the report must not claim an update that never happened
    assert result.optimizer_step is False
    trainer.close()


def test_fit_drops_a_partial_accumulation_window():
    """The tail is DROPPED, not flushed -- a flush cannot be made correct.

    Those microbatches were accumulated under no_sync(), so with dp_size>1 their
    gradients were never reduced (a synced backward is what materialises the
    reduction, and there is no forward left to re-run), and they are scaled k/G
    where the window mean is k/k.  The old flush therefore diverged DP replicas
    and mis-scaled the update -- and with nothing accumulated at all it advanced
    the LR schedule and optimizer_step for an update that never happened.
    """
    from aitrainer import Trainer

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(model, optimizer, scheduler=scheduler,
                      config=FrameworkConfig(device="cpu", grad_accumulation_steps=4,
                                             grad_clip_norm=0.01),
                      loss_fn=lambda output, batch: output.pow(2).sum() * 1000.0)
    before = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    lr_before = scheduler.get_last_lr()[0]

    trainer.fit([(torch.randn(4, 4),) for _ in range(6)])   # one full window + a 2-batch tail

    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    assert trainer.optimizer_step == 1                      # the tail did not step
    assert scheduler.get_last_lr()[0] != lr_before          # advanced once, for the real step
    assert float((after - before).norm()) < 0.1             # the real step was still clipped
    assert all(p.grad is None for p in model.parameters())  # tail gradients discarded
    trainer.close()


def test_fit_discards_the_tail_after_a_failed_step():
    """Residual .grad would be double-counted if a caller retried the window."""
    from aitrainer import Trainer

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                      config=FrameworkConfig(device="cpu", grad_accumulation_steps=4),
                      loss_fn=lambda output, batch: output.sum())

    calls = {"n": 0}

    def flaky(output, batch):                                # fails on the 2nd microbatch
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bad batch")
        return output.pow(2).sum()

    trainer.loss_fn = flaky
    data = [(torch.randn(2, 4),) for _ in range(4)]
    with pytest.raises(RuntimeError):
        trainer.fit(data)
    assert all(p.grad is None for p in model.parameters()), "partial window left in .grad"
    trainer.close()


# --- no_sync wiring and the multi-rank guard --------------------------------

def test_no_sync_is_entered_for_every_non_boundary_microbatch():
    """no_sync() was never called, so each microbatch paid a full gradient reduction."""
    from contextlib import contextmanager

    from aitrainer import Trainer

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.entered = 0

        def forward(self, value): return self.lin(value)

        @contextmanager
        def no_sync(self):
            self.entered += 1
            yield

    model = Recorder()
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                      config=FrameworkConfig(device="cpu", grad_accumulation_steps=3),
                      loss_fn=lambda output, batch: output.pow(2).sum())
    for _ in range(3):
        trainer.train_step((torch.randn(2, 2),))
    # two non-boundary microbatches suppress sync; the third is the boundary
    assert model.entered == 2
    trainer.close()


def test_multi_rank_guard_rejects_a_wrapper_that_cannot_synchronise():
    """DistributedModel.no_sync() degrades to nullcontext(), so guarding on the
    wrapper let an entirely unsynchronised model train with diverging replicas."""
    import dataclasses

    from aitrainer import Trainer
    from aitrainer.distributed_model import DistributedModel
    from aitrainer.runtime import Runtime

    def trainer_for(model):
        runtime = Runtime(device="cpu", seed=0)
        runtime.state = dataclasses.replace(runtime.state, world_size=2)
        # fsdp enabled: dp_size>1 without it is rejected earlier by the capability
        # check, and this test is about the no_sync guard specifically
        config = FrameworkConfig(device="cpu", parallel=ParallelConfig(dp_size=2),
                                 fsdp=FSDPConfig(enabled=True))
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                       config=config, runtime=runtime)

    plain = DistributedModel(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="no_sync"):
        trainer_for(plain)

    class Syncs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, value): return self.lin(value)

        def no_sync(self):
            from contextlib import nullcontext
            return nullcontext()

    trainer_for(DistributedModel(Syncs())).close()          # must not raise


def test_tensor_parallel_without_dp_needs_no_gradient_sync():
    """`tp` is marked STABLE but Trainer refused every tp-only config.

    TP already reduces gradients inside the autograd collectives, and with
    dp_size == 1 there are no replicas to diverge, so no no_sync is required.
    """
    import dataclasses

    from aitrainer import Trainer
    from aitrainer.distributed_model import DistributedModel
    from aitrainer.runtime import Runtime

    runtime = Runtime(device="cpu", seed=0)
    runtime.state = dataclasses.replace(runtime.state, world_size=2)
    # tp_size=2, dp_size=1 -> product 2 == world 2, and no replication
    config = FrameworkConfig(device="cpu", parallel=ParallelConfig(tp_size=2))
    bare = DistributedModel(torch.nn.Linear(2, 2))          # what parallelize returns for TP-only
    trainer = Trainer(bare, torch.optim.SGD(bare.parameters(), lr=0.1),
                      config=config, runtime=runtime)
    trainer.close()


# --- config validation and overlap budgets ----------------------------------

def test_mixed_precision_dtype_is_validated():
    """It was the one dtype field never checked, while the checked ones were inert."""
    from aitrainer.config import ConfigurationError

    for bad in ("float64", object()):
        config = FrameworkConfig(fsdp=FSDPConfig(
            enabled=True, mixed_precision=MixedPrecisionConfig(enabled=True, dtype=bad)))
        with pytest.raises(ConfigurationError):
            config.validate(1)
    FrameworkConfig(fsdp=FSDPConfig(
        enabled=True,
        mixed_precision=MixedPrecisionConfig(enabled=True, dtype="bfloat16"))).validate(1)


@pytest.mark.parametrize("payload", [{"parallel": None}, {"parallel": {"tp_sizee": 2}}, {"parallell": {}}])
def test_from_dict_reports_configuration_errors_not_raw_exceptions(payload):
    """Nested typos and an explicit null leaked bare TypeError/AttributeError."""
    from aitrainer.config import ConfigurationError

    with pytest.raises(ConfigurationError):
        FrameworkConfig.from_dict(payload)


def test_overlap_budget_bounds_inflight_ops():
    """_enforce_budget waited the oldest op but never released it, so the same
    already-WAITED op was re-waited forever and neither budget ever blocked again."""
    from aitrainer.core.lifecycle import AsyncOp
    from aitrainer.overlap.controller import OverlapController

    controller = OverlapController(max_inflight_ops=2)
    for index in range(6):
        op = AsyncOp(f"op{index}")
        controller.submit_op(op)
        controller.wait(op)
    assert len(controller.scheduler.pending) <= 2


# --- checkpoint integrity ---------------------------------------------------

def _converted(tmp_path, *, stage_assignment=None, pp_size=1, tp_size=1, dp_size=1, world=1):
    from aitrainer.checkpoint import CheckpointConverter

    source = tmp_path / "dense.pt"
    torch.save({"model": {"weight": torch.arange(16).reshape(8, 2),
                          "bias": torch.ones(8)}}, source)
    destination = tmp_path / "sharded"
    manifest = CheckpointConverter(partition_dims={"weight": 0}).convert(
        source, destination, world_size=world, dp_size=dp_size, tp_size=tp_size, pp_size=pp_size,
        stage_assignment=stage_assignment)
    return destination, manifest


def test_converter_records_checksums_and_reader_verifies_them(tmp_path):
    """ShardSpec.checksum was never written, so verify_checksum had nothing to check."""
    from aitrainer.checkpoint.reader import ModelLoader

    destination, manifest = _converted(tmp_path, world=2, tp_size=2)
    spec = manifest.tensors["weight"][0]
    assert spec.checksum, "converter must record a checksum for each shard file"

    shard = destination / spec.source_file
    pristine = shard.read_bytes()
    shard.write_bytes(pristine + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ModelLoader().load_for_rank(destination / "manifest.json",
                                    {"weight": torch.zeros(4, 2), "bias": torch.zeros(8)},
                                    world_size=2, logical_sharding={"tp": 2, "pp": 1, "dp": 1})
    shard.write_bytes(pristine)


def test_load_for_rank_rejects_a_mismatched_layout(tmp_path):
    """Comparing only the world-size total accepted tp=2,dp=1 into a tp=1,dp=2 mesh."""
    from aitrainer.checkpoint.reader import ModelLoader

    destination, _ = _converted(tmp_path, world=2, tp_size=2)
    with pytest.raises(ValueError, match="tp="):
        ModelLoader().load_for_rank(destination / "manifest.json",
                                    {"weight": torch.zeros(4, 2), "bias": torch.zeros(8)},
                                    world_size=2, logical_sharding={"tp": 1, "pp": 1, "dp": 2})


def test_save_dcp_can_overwrite_an_existing_checkpoint(tmp_path):
    """A second save raised OSError 66 and the handler then deleted the new one."""
    from aitrainer.checkpoint import CheckpointManager

    target = tmp_path / "dcp"
    manager = CheckpointManager()
    manager.save_dcp(target, state={"w": torch.randn(4, 4)}, metadata={"step": 1})
    manager.save_dcp(target, state={"w": torch.randn(4, 4)}, metadata={"step": 2})
    assert (target / "READY").is_file()
    state = {"w": torch.zeros(4, 4)}
    manager.load_dcp(target, state=state)
    assert state["w"].abs().sum() > 0                       # payload actually replaced


# --- overlap accounting, lifecycle, kernels, activation hooks ---------------

def test_profiler_overlap_ratio_is_bounded():
    """The numerator was communication time and the denominator section time."""
    from aitrainer.profiler import Profiler

    profiler = Profiler()
    profiler.record_wait(4.0, overlapped_seconds=4.0)
    profiler.record_wait(4.0, overlapped_seconds=1.0)
    with profiler.section("tiny"):
        pass
    summary = profiler.summary()
    assert 0.0 <= summary["overlap_ratio"] <= 1.0
    # ratio = hidden / (exposed wait + hidden) = 5 / (8 + 5)
    assert summary["overlap_ratio"] == pytest.approx(5.0 / 13.0)


def test_released_without_completing_is_not_a_silent_success():
    """release() is legal from TIMED_OUT, and re-waiting used to return None."""
    import time

    from aitrainer.core.lifecycle import AsyncOp, LifecycleError

    op = AsyncOp("slow", _wait_fn=lambda _: time.sleep(0.3))
    op.submit()
    time.sleep(0.02)                       # age the op past its deadline
    with pytest.raises(TimeoutError):
        op.wait(timeout=0.01)
    op.release()
    with pytest.raises(LifecycleError, match="without completing"):
        op.wait()

    done = AsyncOp("ok", _wait_fn=lambda _: 7)
    done.submit()
    done.wait()
    done.release()
    assert done.wait() == 7                # a completed op still returns its value


def test_eager_fallback_is_selected_and_not_registerable():
    """attention_capability() reported backend='eager', which register() refuses."""
    from aitrainer.kernels.attention import attention_capability, attention_kernel_backend
    from aitrainer.kernels.backend import KernelBackend

    capability = attention_capability()
    assert attention_kernel_backend() is None          # no flash on a CPU box
    assert capability.backend == "eager"

    backend = KernelBackend("toy", eager=lambda value: value + 1)
    with pytest.raises(ValueError, match="fallback"):
        backend.register("eager", lambda value: value, capability)
    selection = backend.select(device="cpu", dtype="float32")
    assert selection.backend == "eager" and selection.used_fallback


def test_overlap_peak_is_a_concurrency_high_water_mark():
    """inflight_peak reported the number of recorded operations, not a peak."""
    from aitrainer.core.lifecycle import AsyncOp
    from aitrainer.overlap.controller import OverlapController
    from aitrainer.overlap.metrics import OverlapMetrics

    controller = OverlapController()
    for index in range(100):
        op = AsyncOp(f"op{index}")
        controller.submit_op(op)
        controller.wait(op)
    assert controller.summary()["inflight_peak"] <= 2

    metrics = OverlapMetrics()
    assert metrics.summary()["prefetch_hit"] == 0       # honest default, not a claim
    metrics.record_prefetch(hits=5, misses=2)
    assert metrics.summary()["prefetch_hit"] == 5


def test_rms_norm_stays_within_its_declared_error_budget():
    """error_budget=0.0 was unattainable: bf16 rounds the product on its own."""
    from aitrainer.kernels.backend import KernelBackend
    from aitrainer.kernels.fused import rms_norm

    torch.manual_seed(0)
    value = torch.randn(64, 128)
    for dtype in (torch.bfloat16, torch.float16):
        as_dtype = value.to(dtype)
        got = rms_norm(as_dtype, torch.ones(128).to(dtype))
        # Reference on the SAME dtype-cast input, so this measures the kernel's own
        # error rather than the input's quantisation to bf16.
        exact = as_dtype.float()
        reference = exact * torch.rsqrt(exact.pow(2).mean(-1, keepdim=True) + 1e-6)
        relative = ((got.float() - reference).abs() / (reference.abs() + 1e-6)).max().item()
        assert relative <= 2 ** -8, f"{dtype}: {relative}"

    fallback = KernelBackend("toy", eager=lambda value: value).select(device="cpu")
    assert fallback.capability.error_budget > 0          # no longer an impossible 0.0


def test_activation_unpack_is_repeatable_and_fails_loudly_after_drain():
    """A second unpack returned None, and drain() left a dangling pool handle."""
    from aitrainer.offload.activation import (
        ActivationOffloader,
        ActivationOffloadError,
        _SavedTensor,
    )

    offloader = ActivationOffloader(threshold_bytes=1 << 20, keep_last=0)
    source = torch.randn(4, 4)
    record = _SavedTensor(tensor=source, device=source.device, dtype=source.dtype,
                          shape=(4, 4), stride=source.stride())
    record.cpu = source.clone()
    record.tensor = None                                  # as _offload leaves it

    first, second = offloader._unpack(record), offloader._unpack(record)
    assert torch.equal(first, second)                     # was None on the 2nd call

    offloader._records.append(record)
    offloader.drain()
    assert record.cpu is None and record.pooled is False  # no dangling lease
    with pytest.raises(ActivationOffloadError, match="drained"):
        offloader._unpack(record)


# --- shipped presets must actually run --------------------------------------

@pytest.mark.parametrize("preset", ["single_gpu", "performance", "memory_saving"])
def test_shipped_presets_complete_a_step(preset):
    """`memory_saving` used to raise PrecisionError on the very first step.

    It set ``grad_dtype="bfloat16"`` while ``param_dtype`` is inert (parameters
    stay fp32), and ``cast_gradients`` requires the gradient and parameter dtypes
    to match -- so the preset could never complete an update.
    """
    import math
    import warnings as _warnings

    from aitrainer import Trainer
    from aitrainer.presets import ConfigPreset

    model = torch.nn.Linear(4, 2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)   # bf16-on-CPU advisory
        trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3),
                          config=getattr(ConfigPreset, preset)(),
                          loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(
                              output, batch[1]))
        history = trainer.fit([(torch.randn(4, 4), torch.randint(0, 2, (4,)))],
                              epochs=1, max_steps=1)
    assert math.isfinite(history[-1].loss)
    trainer.close()


def test_the_two_dtype_presets_now_differ_on_a_live_field():
    """They used to be byte-identical, so comparing them measured nothing."""
    from aitrainer.presets import ConfigPreset

    saving = ConfigPreset.memory_saving().precision
    performance = ConfigPreset.performance().precision
    assert saving != performance
    assert saving.optimizer_dtype != performance.optimizer_dtype
    assert saving.grad_dtype == performance.grad_dtype == "float32"


def test_cast_optimizer_state_preserves_the_step_counter():
    """Adam keeps `step` as a float tensor; casting it broke bias correction."""
    from aitrainer.precision import cast_optimizer_state

    weight = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.AdamW([weight], lr=1e-3)
    (weight * 2).sum().backward()
    optimizer.step()
    cast_optimizer_state(optimizer, "bfloat16")
    state = optimizer.state[weight]
    assert state["exp_avg"].dtype is torch.bfloat16
    assert state["step"].dtype is torch.float32


def test_bf16_on_cpu_warns_that_it_has_no_effect():
    """On CPU a reduced-precision compute dtype silently runs fp32."""
    from aitrainer import Trainer
    from aitrainer.config import PrecisionConfig

    model = torch.nn.Linear(2, 2)
    with pytest.warns(RuntimeWarning, match="has no effect"):
        Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                config=FrameworkConfig(device="cpu",
                                       precision=PrecisionConfig(compute_dtype="bfloat16")))


def test_the_documented_quickstart_example_is_reproducible(tmp_path):
    """`examples/tiny_transformer` calls itself reproducible and sets seed=123.

    It built the model BEFORE `Trainer(...)` -- and the seed is applied when the
    runtime is constructed inside `Trainer.__init__` -- so weight init came from
    the ambient RNG and every run reported a different loss.  The data generator
    was seeded explicitly, which made the example look seed-controlled.
    Runs the shipped script twice in a throwaway cwd and compares the loss.
    """
    import re
    import subprocess
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples" / "tiny_transformer" / "train.py"
    losses = []
    for _ in range(2):
        completed = subprocess.run([sys.executable, str(example)], cwd=tmp_path,
                                   capture_output=True, text=True, timeout=300, check=False)
        assert completed.returncode == 0, completed.stderr[-400:]
        match = re.search(r"loss=([0-9.]+)", completed.stdout)
        assert match, completed.stdout
        losses.append(match.group(1))
    assert losses[0] == losses[1], f"the example is not reproducible: {losses}"


# --- end-to-end accumulation invariant --------------------------------------

def _seeded_linear() -> torch.nn.Module:
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(4, 3))
    with torch.no_grad():
        module[0].weight.fill_(0.1)
        module[0].bias.fill_(0.0)
    return module


def test_accumulation_window_equals_one_step_over_the_whole_batch():
    """Guards the whole accumulation path at once.

    A mean-reduced loss over N microbatches must produce exactly the gradient of
    one step over the concatenated batch: any error in the ``/grad_accumulation_steps``
    division, the no_sync boundary timing, or the step-count would show up here.
    """
    from aitrainer import Trainer

    torch.manual_seed(1)
    inputs, labels = torch.randn(4, 4), torch.randint(0, 3, (4,))
    loss_fn = lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1])

    whole = _seeded_linear()
    whole_trainer = Trainer(whole, torch.optim.SGD(whole.parameters(), lr=0.1),
                            config=FrameworkConfig(device="cpu"), loss_fn=loss_fn)
    whole_trainer.train_step((inputs, labels))

    split = _seeded_linear()
    split_trainer = Trainer(split, torch.optim.SGD(split.parameters(), lr=0.1),
                            config=FrameworkConfig(device="cpu", grad_accumulation_steps=4),
                            loss_fn=loss_fn)
    for index in range(4):
        split_trainer.train_step((inputs[index:index + 1], labels[index:index + 1]))

    assert split_trainer.optimizer_step == 1
    for expected, actual in zip(whole.parameters(), split.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    whole_trainer.close()
    split_trainer.close()


def test_an_incomplete_accumulation_window_does_not_update():
    from aitrainer import Trainer

    torch.manual_seed(1)
    inputs, labels = torch.randn(4, 4), torch.randint(0, 3, (4,))
    model = _seeded_linear()
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                      config=FrameworkConfig(device="cpu", grad_accumulation_steps=4),
                      loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1]))
    before = [p.detach().clone() for p in model.parameters()]
    for index in range(3):                                   # one short of the window
        trainer.train_step((inputs[index:index + 1], labels[index:index + 1]))
    assert trainer.optimizer_step == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    trainer.close()


# --- publish robustness, per-stage loading, group guards, buffer pool -------

def _save_small(manager, path):
    manager.save(path, model={"w": torch.nn.Linear(2, 2)}, optimizer=None, config={})


def test_saving_onto_a_file_path_works_and_does_not_poison_it(tmp_path):
    """rmtree on a displaced FILE left `.NAME.previous`, failing every later save."""
    from aitrainer.checkpoint import CheckpointManager

    target = tmp_path / "ckpt"
    target.write_text("stale single-file checkpoint")
    manager = CheckpointManager()
    _save_small(manager, target)
    assert target.is_dir()
    assert not (tmp_path / ".ckpt.previous").exists()
    _save_small(manager, target)                     # must not be poisoned


def test_saving_through_a_symlink_keeps_the_link(tmp_path):
    """The `latest -> step-N` layout must not be replaced by a real directory."""
    import os

    from aitrainer.checkpoint import CheckpointManager

    manager = CheckpointManager()
    real = tmp_path / "step-1000"
    _save_small(manager, real)
    link = tmp_path / "latest"
    os.symlink(real, link)
    _save_small(manager, link)
    assert link.is_symlink()
    assert real.is_dir()


def _converted_stages(tmp_path):
    from aitrainer.checkpoint import CheckpointConverter

    source = tmp_path / "dense.pt"
    torch.save({"l0.weight": torch.randn(8, 4), "l1.weight": torch.randn(4, 8)}, source)
    destination = tmp_path / "conv"
    CheckpointConverter(partition_dims={"l0.weight": 0, "l1.weight": 0}).convert(
        source, destination, pp_size=2, tp_size=1, dp_size=1,
        stage_assignment={"l0.weight": 0, "l1.weight": 1})
    return destination


def test_a_stage_local_model_skips_tensors_owned_by_other_stages(tmp_path):
    """Raising on an unmatched shard made per-stage PP loading impossible."""
    from aitrainer.checkpoint.reader import ModelLoader

    destination = _converted_stages(tmp_path)
    stage_one = {"l1.weight": torch.zeros(4, 8)}                 # stage 1 owns only l1
    stats = ModelLoader().load_for_rank(destination / "manifest.json", stage_one,
                                        pp_rank=1, tp_rank=0, dp_rank=0)
    assert stats["tensors_loaded"] == 1
    assert stage_one["l1.weight"].abs().sum() > 0                # real values, not zeros


def test_a_tensor_the_model_owns_with_no_shard_still_raises(tmp_path):
    """The loud failure must survive for the case it was added for."""
    from aitrainer.checkpoint.format import Manifest, ShardSpec, write_manifest
    from aitrainer.checkpoint.reader import ModelLoader

    destination = tmp_path / "m"
    destination.mkdir()
    # only a tp_rank=1 shard exists, but this rank is tp_rank=0
    write_manifest(Manifest(world_size_at_save=2, logical_sharding={"tp": 2, "pp": 1, "dp": 1},
                            tensors={"w": (ShardSpec("w", (4, 4), "float32", "tp_sharded",
                                                     target_rank=(0, 1, 0),
                                                     source_file="rank_00001.pt", source_key="w",
                                                     local_shape=(2, 4)),)}),
                   destination / "manifest.json")
    torch.save({"w": torch.zeros(2, 4)}, destination / "rank_00001.pt")
    with pytest.raises(KeyError, match="no shard"):
        ModelLoader().load_for_rank(destination / "manifest.json", {"w": torch.zeros(2, 4)},
                                    pp_rank=0, tp_rank=0, dp_rank=0)


def test_a_single_rank_mapping_in_a_larger_world_is_rejected(monkeypatch):
    """It returned None groups, and torch reads a None group as ALL ranks."""
    import torch.distributed as dist

    from aitrainer.parallel.groups import ProcessGroupError, ProcessGroups
    from aitrainer.topology import make_rank_mapping

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    with pytest.raises(ProcessGroupError, match="does not match"):
        ProcessGroups.create(make_rank_mapping(1))


def test_device_mesh_manager_honours_an_explicit_process_group(monkeypatch):
    """The parameter was accepted and then silently ignored."""
    import torch.distributed as dist

    from aitrainer.mesh import DeviceMeshManager

    seen: list[object] = []
    sentinel = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size",
                        lambda group=None: (seen.append(group), 4)[1])
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "new_group", lambda ranks, **kwargs: ("h", tuple(ranks)))

    # A non-identity permutation deliberately skips init_device_mesh (it assumes
    # logical rank order), which would otherwise need a real default group.
    manager = DeviceMeshManager(pp_size=1, dp_size=2, tp_size=2, process_group=sentinel,
                                global_ranks=(1, 0, 3, 2))
    assert seen and seen[0] is sentinel          # the supplied group was consulted
    assert manager.mapping.world_size == 4


def test_buffer_pool_rejects_strided_layout_and_resolves_string_dtypes():
    """empty_strided(shape, shape) passed the shape as strides: aliased + quota bypass."""
    from aitrainer.memory import BufferKey, BufferPoolError, PinnedBufferPool

    pool = PinnedBufferPool(1 << 20, pin_memory=False)
    with pytest.raises(BufferPoolError, match="strided"):
        pool.acquire(BufferKey((4, 8), torch.float32, layout="strided"))
    assert pool.acquire(BufferKey((4,), "float32")).dtype is torch.float32
