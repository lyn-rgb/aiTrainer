"""Real multi-process checks on a live Gloo process group.

Every other "distributed" test in this repository runs in a single process and
checks shapes at ``world_size == 1``.  These launch one OS process per rank, so
the items ``docs/代码审计报告.md`` records as *fixed but not verifiable without a
real multi-rank job* are exercised for real:

* §4.10 a -- canonical ``new_group`` ordering, with the removed ordering kept as
  a negative control that must still deadlock;
* §4.1 -- ``no_sync`` accumulation across replicas, and the FSDP2 properties that
  replaced it (accumulation window, global gradient norm);
* tensor and sequence parallelism against dense references, including the
  compounded DP+TP(+SP) axes and the layouts those produce;
* checkpointing at world_size > 1: the per-rank format, the opt-in DCP format on
  every axis, and resharding a sharded checkpoint into a single process;
* §4.10 d -- the variable-length Ulysses refusal.

``torchrun`` cannot start here (its rendezvous resolves the hostname through
mDNS and dies with ``gai error: 8``), but that is a property of ``torchrun``,
not of multi-rank: setting the rendezvous variables by hand works.  Hence
``tests/multirank/harness.py``.  A module-level ``importorskip`` would hide this
whole file, so each test asserts on a ``RESULT`` payload rather than on an exit
code alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.multirank.harness import require_success, run_case

TOLERANCE = 1e-6


def same_weights(left, right, what: str) -> None:
    """Assert two runs produced the same weights, to ``TOLERANCE`` and not bit-for-bit.

    The distinction this draws is the whole point.  A weight that differs by a
    few ULP is not a divergence -- it is the same training computed through a
    different summation order, and ``==`` on a list of float32 values reports it
    as a failure.  Measured on the Linux runner: two configurations that agree
    to the last bit on the development host differed by ``1.9e-09`` on a value
    of ``0.02``, which is about one ULP of float32 (eps 1.19e-07).  A real
    divergence is O(value), so 1e-6 still catches every one of them.

    This is not a relaxation introduced to quiet a red build.  The file already
    made this comparison with a tolerance in
    ``test_checkpoint_roundtrip_at_two_ranks``' sibling, for exactly this reason
    and with this exact constant; the assertions converted here were the
    inconsistency, not the rule.

    Discrete values stay exact: ``applied`` is a list of booleans and
    ``global_step`` an int, and neither has a rounding story to tell.
    """
    assert len(left) == len(right), (
        f"{what}: {len(left)} values against {len(right)} -- not the same set of tensors")
    worst = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    assert worst < TOLERANCE, (
        f"{what}: max|diff| = {worst:.3e} exceeds {TOLERANCE:g}. A few ULP is "
        f"rounding; this is a real divergence.")


@pytest.mark.parametrize("shape", ["2,2,1", "2,1,2", "1,2,2"])
def test_group_creation_membership_at_four_ranks(shape):
    """All ranks must create identical group sequences and get their own groups."""
    results = require_success(run_case("groups_canonical_order", 4, shape=shape,
                                       hard_timeout=60.0))
    assert len(results) == 4
    for payload in results:
        assert payload["plan_len"] == 8, "every rank must issue the same number of new_group calls"

    # The worker checks each rank's group against the mapping.  Here we assert
    # the property the mapping cannot express on its own: the four ranks must
    # agree with one another, and the groups of an axis must partition the world
    # (a rank that built a wrong-sized group would break one of these).
    for axis in ("pp", "dp", "tp"):
        memberships = [tuple(sorted(results[r]["membership"][axis])) for r in range(4)]
        covered = sorted({rank for group in memberships for rank in group})
        assert covered == [0, 1, 2, 3], f"{axis} groups do not cover the world: {memberships}"
        for rank, group in enumerate(memberships):
            assert rank in group, f"rank {rank} is missing from its own {axis} group"
        # Ranks in the same group must report identical membership.
        for left in range(4):
            for right in range(4):
                if left in memberships[right]:
                    assert memberships[left] == memberships[right], (
                        f"ranks {left} and {right} disagree about their {axis} group")


def test_group_creation_order_is_load_bearing():
    """The pre-fix per-rank ordering must still hang, or the fix proves nothing.

    This is the counterfactual for the test above.  ``new_group`` must be called
    by every rank in the same order with the same rank list; creating only a
    rank's own groups makes the per-process call counter carry mismatched rank
    lists at the same store index, and no rank observes the expected barrier.
    """
    results = run_case("groups_legacy_order_deadlocks", 4, shape="2,2,1",
                       timeout_seconds=3.0, hard_timeout=10.0,
                       retry_on_rendezvous=False)
    # The worker already distinguishes the two outcomes -- it returns a "did not
    # block" note when the control does not reproduce -- so the failure message
    # carries what each rank actually did.  Without it the assertion says only
    # that something completed, which is the part that was already known.
    observed = [r.result for r in results if r.result] or [r.describe() for r in results]
    assert not all(result.returncode == 0 for result in results), (
        "the removed ordering completed cleanly, so the canonical ordering fix "
        "is not what makes group creation terminate. What the ranks reported: "
        f"{json.dumps(observed, sort_keys=True)}")
    combined = " ".join((result.error or "") + result.stderr for result in results)
    assert "wait timeout" in combined or "TimeoutExpired" in combined, (
        f"expected a store-barrier timeout from the legacy ordering, got:\n{combined[-2000:]}")


def test_tensor_parallel_matches_dense_and_stays_dtensor():
    """Two claims: the sharded numbers are right, and the sharding is *represented*.

    The first is ordinary correctness.  The second is why the hand-written
    layers were replaced: their parameters were plain tensors holding a shard,
    so ``get_model_state_dict`` reported each rank's slice and
    ``torch.distributed.checkpoint`` kept whichever rank wrote last. The global
    shape and the DTensor placement are what make the sharded checkpoint path
    meaningful, and neither shows up in a numeric comparison.
    """
    for payload in require_success(run_case("tensor_parallel_matches_dense", 2, hard_timeout=120.0)):
        assert payload["forward_error"] < TOLERANCE
        assert payload["input_grad_error"] < TOLERANCE
        for name, error in payload["parameter_errors"].items():
            assert error < TOLERANCE, f"{name} diverged from the dense reference"
        assert payload["weight_type"] == "DTensor", (
            "the parameter came back as a plain tensor, so nothing records that it is a "
            "slice of a larger logical tensor")
        assert "Shard" in payload["weight_placements"]
        assert payload["weight_global_shape"] == payload["expected_global_shape"], (
            "the DTensor must report the GLOBAL shape; a local one is the old defect")
        assert payload["weight_local_shape"][0] * 2 == payload["weight_global_shape"][0], (
            "the local shard is not half the global weight at tp=2")


def test_a_plan_that_names_nothing_is_refused():
    """``tp_size>1`` over an unmatched model must raise, not replicate silently.

    Measured before the guard: a plain ``nn.Sequential`` at tp_size=2 kept its
    full parameter count on every rank and printed nothing -- N ranks training N
    full copies on N slices of the data.  Asserting the *message* matters too,
    because the plausible failure is an error the reader cannot act on.
    """
    for payload in require_success(run_case("tensor_parallel_refuses_an_unmatched_plan", 2,
                                            hard_timeout=60.0)):
        assert payload["refused"] is True, (
            "an unmatched plan was accepted, so the model was silently replicated")
        assert "names no module to shard" in payload["message"]
        assert "tp_size=1" in payload["message"], "the error must say how to proceed"


@pytest.mark.parametrize(("world", "options"), [
    (2, {"dp": 2, "tp": 1}),
    (2, {"dp": 1, "tp": 2}),
    (2, {"dp": 1, "tp": 2, "sp": "megatron"}),
    (4, {"dp": 2, "tp": 2}),
])
def test_compounded_axes_match_dense(world, options):
    """``parallelize`` must equal dense for each axis combination it accepts.

    This parameterization is the regression test for a composition that was
    broken and unobserved: after TP parameters became DTensors, ``fully_shard``
    rejected the DP mesh ``wrap_fsdp`` built (``DeviceMesh.from_group`` has no
    parent; FSDP2 wants DP and TP to share one), so ``tp_size=2`` with
    ``fsdp.enabled`` failed at construction while ``dp_size=2`` alone worked.

    Every rank gets identical data here, which makes the single-process run a
    valid reference -- a DP reduction over identical replicas is the identity on
    the gradient.  That also means this case cannot see *divergent* replicas;
    ``test_trainer_fit_over_fsdp_matches_single_process`` and the clipping cases
    use rank-distinct data for that.
    """
    payloads = require_success(run_case("composition_matches_dense", world, options=options,
                                        hard_timeout=180.0))
    assert len(payloads) == world
    for payload in payloads:
        assert payload["forward_error"] < TOLERANCE
        assert payload["input_grad_error"] < TOLERANCE
        for name, error in payload["parameter_errors"].items():
            assert error < TOLERANCE, f"{name} diverged from the dense reference"
        assert payload["weight_type"] == "DTensor", (
            "the composed axes must still leave the parameters representable")
    if options.get("tp", 1) > 1 and options.get("dp", 1) > 1:
        placements = payloads[0]["weight_placements"]
        assert "Shard" in placements and "," in placements, (
            f"dp=2/tp=2 must shard on BOTH axes; got {placements}")


def test_sequence_parallel_matches_dense():
    """SP inside the norms: the numbers, and the sharding the numbers cannot show.

    The input gradient is the sharp edge.  An earlier version of the style took
    the local ``chunk`` by hand; a chunk's backward hands each rank only its own
    slice's contribution, so a *replicated* input ends up with a partially
    filled gradient -- 4.8e-03 and 7.0e-03 against 9.3e-10 here.  Expressing the
    scatter as a ``Replicate -> Shard`` transition instead lets DTensor's
    autograd supply the all-gather that the hand-written ``gather_sequence``
    used to do.
    """
    _assert_sequence_parallel_numbers(require_success(
        run_case("sequence_parallel_matches_dense", 2, hard_timeout=120.0)))


def test_sequence_parallel_covers_rmsnorm_too():
    """``nn.RMSNorm`` is what a Llama uses, and the style has to cover it.

    Before this, it did not: the style matched ``torch.nn.LayerNorm`` only, so a
    Llama-shaped model at ``tp_size=2`` with ``sp_backend='megatron'`` sharded
    nothing while reporting success.  That is now refused rather than ignored
    (``require_sequence_parallel_targets``), and this case is the other half:
    the type list was extended instead of only guarded, because the style is
    indifferent to which norm it wraps.  It shards the **sequence** axis and
    leaves the norm its own axis -- the last one -- whole on every rank, so
    anything that normalises over the hidden dimension applies.

    Running the SAME assertions as the LayerNorm case is the point.  The numbers
    alone would not have been enough: a style that sharded nothing matches dense
    exactly, which is why the shared body also asserts the activation is a
    ``Shard`` DTensor of half the sequence, and that the optimizer's momentum
    ends up replicated.
    """
    _assert_sequence_parallel_numbers(require_success(
        run_case("sequence_parallel_matches_dense", 2, options={"norm": "rmsnorm"},
                 hard_timeout=120.0)))


def _assert_sequence_parallel_numbers(payloads) -> None:
    """Every claim the SP cases make, applied to whichever norm type they ran.

    Shared between the LayerNorm and RMSNorm cases so the two cannot drift
    apart: the RMSNorm case exists to answer "does the style cover this norm
    type", and that is only answered if it is asked with the same assertions.
    """
    for payload in payloads:
        assert payload["forward_error"] < TOLERANCE
        assert payload["input_grad_error"] < TOLERANCE, (
            "the input gradient through the sharded norm is wrong; this is what a "
            "hand-rolled scatter loses")
        assert payload["weight_grad_error"] < TOLERANCE, (
            "the replicated norm parameters did not receive the reduced gradient")
        # None when the norm has no bias at all (nn.RMSNorm is weight-only), so
        # this is a skip when the parameter does not exist, not a pass.
        if payload["bias_grad_error"] is not None:
            assert payload["bias_grad_error"] < TOLERANCE

        # The error columns above are all computed through ``full_tensor()``,
        # which ALL-REDUCES a Partial placement -- so a norm gradient that was
        # never reduced compares EQUAL to the dense reference and every
        # assertion above passes anyway.  Measured: with the reduction removed
        # this whole test still passed, while SGD's momentum buffer came back
        # ``(Partial(sum),)`` holding each rank's own partial sum, and a
        # checkpoint written from it resumed 2.08e-01 away from the run that
        # wrote it.
        #
        # These are the assertions that go red instead.  First the cause:
        assert "Partial" in payload["raw_grad_placements"], (
            "a backward over the sequence-sharded norm should leave the "
            "replicated parameters holding a per-rank partial sum; if this is "
            "no longer true, torch has started reducing it and "
            "reduce_replicated_gradients is doing a second all-reduce for "
            f"nothing.  Got {payload['raw_grad_placements']!r}")

        # Then the consequence, measured through the path that trains: the
        # momentum the optimizer actually stores must be Replicate and must
        # agree rank to rank.  It is the optimizer's state, not the parameter
        # update, that a sharded checkpoint carries -- the update comes out
        # right either way, because DTensor reduces when it computes the new
        # parameter value.
        assert "Replicate" in payload["momentum_placements"], (
            "the optimizer's momentum buffer is not replicated, so each rank "
            "stores a different partial sum; the update still looks right, but "
            "gradient clipping and every checkpoint of this state are wrong.  "
            f"Got {payload['momentum_placements']!r}")
        locales = [item["momentum_local"] for item in payloads]
        assert all(item == locales[0] for item in locales), (
            "the replicated norm parameters hold different optimizer momentum "
            "on different ranks, so they are not replicated")

        observed = payload["observed"]
        assert observed["type"] == "DTensor", (
            "the activation inside the norm was not a DTensor, so sequence "
            "parallelism sharded nothing")
        assert "Shard" in observed["placements"]
        assert observed["global_shape"][1] == payload["sequence_length"], (
            "the DTensor must describe the GLOBAL sequence length")
        assert observed["local_shape"][1] * 2 == payload["sequence_length"], (
            "each rank holds half the sequence, which is the whole point of SP -- a "
            "style that shards nothing would still match the dense numbers exactly")


def test_pipeline_nested_model_needs_an_execution_order():
    """The pass-through, and the split each rank ends up with.

    ``test_pp_shapes`` covers the split in one process; this covers the wiring,
    which is where a silent failure would live: an ``execution_order`` that
    ``parallelize`` dropped on the floor would look exactly like a caller that
    never passed one, and the two ranks would have to agree about the stages or
    the schedule would hang rather than fail.
    """
    payloads = require_success(run_case("pipeline_needs_execution_order_for_a_nested_model", 2,
                                        hard_timeout=120.0))
    for payload in payloads:
        assert payload["refused"] is True
        assert "defines no forward" in payload["message"], (
            f"refused for the wrong reason: {payload['message']}")
        assert payload["stage_count"] == 2

    # Each rank holds the stage its pp coordinate says, and the layer stack is
    # actually divided rather than gathered onto one of them.
    by_stage = {payload["stage_id"]: payload["units"] for payload in payloads}
    assert sorted(by_stage) == [0, 1], "both stages must be represented"
    assert by_stage[0] == ["embed_tokens", "layer_0", "layer_1", "layer_2"]
    assert by_stage[1] == ["layer_3", "norm", "lm_head"]


def test_sequence_parallel_refuses_at_world_two_for_an_unknown_norm():
    """The refusal has to reach the real call path, at every rank.

    The unit test in ``test_sp_single`` checks the guard function; this checks
    that ``parallelize`` actually calls it, and that the model the guard rejects
    is rejected on BOTH ranks rather than on the one that happened to look
    first -- a refusal that is not rank-consistent is a hang, not an error.

    The norm here is the model's own rather than ``nn.RMSNorm``: that one is
    shardable and covered by ``test_sequence_parallel_covers_rmsnorm_too``.
    """
    for payload in require_success(run_case("sequence_parallel_refuses_an_unknown_norm", 2,
                                            hard_timeout=120.0)):
        assert payload["refused"] is True
        assert "no norm the sequence-parallel style can shard" in payload["message"], (
            f"refused for the wrong reason: {payload['message']}")
        # And the same block with a LayerNorm is still sharded, so the guard is
        # about the norm type and not about refusing SP in general.
        assert payload["accepted_norm_weight_type"] == "DTensor"
        assert "Replicate" in payload["accepted_norm_placements"]


def test_variable_length_ulysses_refuses_at_world_two():
    """§4.10 d: the path whose mask is wrong must refuse, not approximate."""
    for payload in require_success(run_case("ulysses_seq_lens_refuses", 2, hard_timeout=60.0)):
        assert payload["refused"] is True
        assert "seq_lens=None" in payload["message"]


def test_ulysses_head_exchange_requires_nccl_not_gloo():
    """Documents a real constraint: Ulysses cannot run on Gloo at all.

    ``all_to_all`` has no Gloo implementation, so every ``world_size > 1``
    Ulysses path is CUDA/NCCL-only.  Nothing in the repository says so, and the
    dense-equivalence check therefore cannot be run on a CPU-only machine.  When
    Gloo gains all-to-all this test starts failing, which is the signal to turn
    on the equivalence check instead.
    """
    payloads = require_success(run_case("ulysses_attention_equivalence", 2, hard_timeout=60.0))
    for payload in payloads:
        assert payload["failed"], (
            f"rank {payload['rank']} ran Ulysses attention on Gloo and returned "
            f"shape {payload.get('shape')} with max|diff| {payload.get('error')} "
            f"against dense SDPA -- the all-to-all did not happen, so this is a "
            f"local computation wearing a success. Environment: "
            f"{json.dumps(payload['facts'], sort_keys=True)}")
        assert "alltoall" in payload["failure"], (
            f"rank {payload['rank']} failed for the wrong reason: {payload['failure']}. "
            f"Environment: {json.dumps(payload['facts'], sort_keys=True)}")


def test_fsdp_initialises_on_a_cpu_only_host():
    """FSDP2 must shard in place with a resolved CPU device.

    FSDP's own device inference resolves the custom 'mps' backend on a CPU-only
    macOS host and raises ``Custom backend 'mps' not implement
    torch.mps.current_device``.  ``wrap_fsdp`` therefore has to forward the
    device the runtime resolved rather than forwarding CUDA only.

    The parameter type is the part that is not a tautology.  ``fully_shard``
    returns the module it mutated, so ``wrap_fsdp`` returning a module proves
    nothing on its own -- a ``wrap_fsdp`` that never called ``fully_shard`` would
    do the same.  DTensor parameters are what sharding actually looks like.

    ``in_place`` and the class name are a pair, not a contradiction: FSDP2 swaps
    the instance's class to a dynamically built ``FSDP<ClassName>`` to install
    its hooks, so the same object answers to a different ``type()``.
    """
    for payload in require_success(run_case("fsdp_cpu_wrap", 2, hard_timeout=60.0)):
        assert payload["in_place"] is True, "FSDP2 mutates in place; it returns no wrapper"
        assert payload["wrapped"] == "FSDPLinear", (
            "fully_shard installs its hooks by replacing the class of the module it "
            "was given")
        assert payload["parameter_types"] == ["DTensor"], (
            "the module came back unsharded: FSDP2 holds parameters as DTensors")


def test_fsdp_gradient_clipping_uses_the_global_norm():
    """Clipping must not depend on which shard a rank happens to hold.

    The near-miss this guards: ``torch.nn.utils.clip_grad_norm_`` is the obvious
    replacement for FSDP1's ``module.clip_grad_norm_``, it runs without error on
    DTensor parameters, and it returns a plausible number -- but that number is
    the local shard's norm, so each rank scales its gradients differently.
    Measured before the fix: 0.551 and 0.395 against a true norm of 0.678.

    Two independent checks, because either alone could pass by accident:

    * the reported norm must equal the norm computed from ``full_tensor()``
      (ground truth, sharing no code with the clipping path);
    * after a real clip the two ranks must hold identical gradients and the
      global norm must be the cap.  Agreement alone would not catch a *uniformly*
      wrong scale, and matching the truth alone would not catch divergence.
    """
    payloads = require_success(run_case("fsdp_grad_norm_is_global", 2, hard_timeout=120.0))
    assert len(payloads) == 2
    for payload in payloads:
        assert payload["measured_norm"] == pytest.approx(payload["true_norm"], rel=1e-5), (
            f"clip_grad_norm_ reported {payload['measured_norm']} but the global norm is "
            f"{payload['true_norm']} -- that is the shard-local norm")
        assert payload["post_clip_norm"] == pytest.approx(payload["cap"], rel=1e-4), (
            "the clip did not land on the cap, so the scaling used the wrong norm")
    assert payloads[0]["gradients"] == payloads[1]["gradients"], (
        "the ranks ended up with different gradients: they clipped by different factors")


def test_fsdp_accumulation_window_suppresses_the_reduction():
    """Stage A's dangerous property: a silent no-op is the only failure mode.

    ``DistributedModel.no_sync`` used to fall back to ``nullcontext`` when the
    module had no ``no_sync``.  FSDP2 renamed that method, so every FSDP2 config
    would have degraded to one full gradient reduction per microbatch -- slower,
    different reduction timing, and no exception anywhere.

    Three assertions, each doing separate work:

    * ``reduced_vs_local`` proves the fixture can tell the two apart.  Both ranks
      receive different data, so a reduced gradient and a rank-local gradient
      differ; with identical data this whole case would pass vacuously.
    * ``materialised_inside_window`` is the direct observation: inside the
      window FSDP2 leaves ``.grad`` as ``None`` (measured -- it accumulates in
      an internal buffer and materialises nothing).  A no-op ``no_sync`` would
      materialise a DTensor on the very first microbatch.
    * ``flush_vs_four_units`` / ``flush_vs_one_unit``: the boundary backward must
      flush *all four* accumulations.  Reducing only the last microbatch would
      still agree with "a reduction happened", so agreeing with ``1 * unit`` is
      the failure this pins down.  (Measured at 0.0 and 2.8e+00 respectively.)
    """
    payloads = require_success(run_case("fsdp_accumulation_window_suppresses_reduction", 2,
                                        hard_timeout=120.0))
    assert len(payloads) == 2
    for payload in payloads:
        assert payload["reduced_vs_local"] > 1e-3, (
            "reduced and rank-local gradients agree, so this case cannot detect "
            "a missing reduction")
        assert payload["materialised_inside_window"] == [False, False, False], (
            "the gradient was materialised inside the accumulation window, so the "
            "reduce-scatter was not suppressed")
        assert payload["flush_vs_one_unit"] > 1e-3, (
            "the boundary backward reduced only the last microbatch; the earlier "
            "accumulated gradients were dropped")
        assert payload["flush_vs_four_units"] < 1e-5 * max(1.0, payload["magnitude"]), (
            "the boundary backward did not flush every accumulated microbatch")


def test_trainer_fit_over_fsdp_matches_single_process():
    """§4.1 end to end: the real ``Trainer``, real FSDP, real DP-group reduction.

    Both ranks are fed identical data, so the reduced gradient equals the
    single-process gradient and the parameters must match a ``world_size == 1``
    run exactly.  Two-process agreement alone would not catch a *consistently*
    wrong reduction or a mis-scaled update; the reference does.

    The batch count is not a multiple of ``grad_accumulation_steps``, so a
    trailing partial window exists and must be dropped rather than flushed --
    re-adding the flush makes the multi-rank run fail outright.  It also
    exercises ``get_model_state_dict``, which replaced FSDP1's
    ``full_state_dict_context`` (whose ``offload_to_cpu=True`` used to segfault
    the process when the parameters were already on CPU).

    Note what this case *cannot* catch: the two ranks are fed identical data,
    so a missing accumulation window (one reduction per microbatch instead of
    one per window) produces the same sum and the same parameters.  That is
    ``test_fsdp_accumulation_window_suppresses_the_reduction``'s job.
    """
    reference = require_success(run_case("trainer_fit_fsdp_accumulation", 1,
                                         hard_timeout=120.0))[0]
    assert reference["global_step"] == 7
    assert reference["applied"] == [False, False, True, False, False, True, False], (
        "only the microbatch that closes a window may run an optimizer step")
    assert reference["optimizer_steps"] == 2, "the trailing partial window must be dropped"

    replicas = require_success(run_case("trainer_fit_fsdp_accumulation", 2, hard_timeout=120.0))
    assert len(replicas) == 2
    for payload in replicas:
        assert payload["applied"] == reference["applied"]
        assert payload["weight"] == reference["weight"], "FSDP replica diverged from the reference"
        assert payload["bias"] == reference["bias"]


def test_trainer_gradient_clipping_matches_single_process():
    """``grad_clip_norm`` end to end, through ``Trainer.fit``, over FSDP2.

    ``test_fsdp_gradient_clipping_uses_the_global_norm`` proves the function is
    right; this proves the production path *calls* it right.  ``run_optimizer_step``
    has to pass the sharded module and read the norm where DTensor gradients
    exist -- no multirank case set ``grad_clip_norm`` before this one, so the
    wiring had never executed under FSDP2 at all.

    The reference is the world_size=1 run of the same case: identical data on
    both ranks means FSDP2's reduced gradient equals the single-process gradient,
    so the global norm, the clip coefficient and every weight must match exactly.
    """
    reference = require_success(run_case("trainer_fit_fsdp_clipping", 1,
                                         hard_timeout=120.0))[0]
    assert reference["optimizer_steps"] == 5
    assert None not in reference["grad_norms"], "clipping is configured but no norm was reported"
    assert max(reference["grad_norms"]) > reference["cap"], (
        "no step's gradient norm exceeded the cap, so this case never actually clipped")

    replicas = require_success(run_case("trainer_fit_fsdp_clipping", 2, hard_timeout=120.0))
    assert len(replicas) == 2
    for payload in replicas:
        assert payload["grad_norms"] == pytest.approx(reference["grad_norms"], rel=1e-5), (
            "the norm FSDP2 reported differs from the single-process one -- a shard-local "
            "norm is strictly smaller than the global norm")
        assert payload["weight"] == reference["weight"], "clipped weights diverged"
        assert payload["bias"] == reference["bias"]


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pipeline_production_path_matches_single_process(schedule):
    """The PP path must produce, at pp_size=2, what one process produces alone.

    This is the first test in the repository to execute
    ``PipelineStage.pipeline_step`` / ``_pipeline_step_1f1b`` at ``pp_size > 1``.
    Neither the 17 in-process protocol tests (they drive
    ``parallel/pp_schedule.py``, which the production path does not call) nor any
    other test reached these lines: every other ``PipelineStage`` construction
    passes ``pp_size=1`` and takes the plain-module shortcut above the schedule.
    That gap is why the pipeline consolidation is staged behind this case.

    Each rank reports only its own stage's parameters, so the two are
    concatenated before comparison -- that concatenation is also what checks the
    stage split put the right layers on the right rank.
    """
    reference = require_success(run_case("pp_pipeline_step", 1,
                                         options={"schedule": schedule},
                                         hard_timeout=120.0))[0]
    replicas = require_success(run_case("pp_pipeline_step", 2,
                                        options={"schedule": schedule},
                                        hard_timeout=120.0))
    assert len(replicas) == 2
    joined = replicas[0]["parameters"] + replicas[1]["parameters"]
    assert len(joined) == len(reference["parameters"]), "the two stages must partition the model"
    same_weights(joined, reference["parameters"], "pipeline parameters")
    same_weights(replicas[1]["losses"], reference["losses"], "last-stage loss")
    assert all(payload["global_step"] == reference["global_step"] for payload in replicas)


def test_pipeline_accumulation_drops_the_trailing_window():
    """A trailing partial window must be dropped on the PP path too.

    ``fit``'s epilogue was the site of a real regression on the non-pipeline path
    (a flush stepped on gradients accumulated under ``no_sync`` and never
    reduced).  This checks the same contract holds when ``pipeline_step`` owns
    the backward.
    """
    options = {"schedule": "gpipe", "accumulation": "3", "steps": "7"}
    reference = require_success(run_case("pp_pipeline_step", 1, options=options,
                                         hard_timeout=120.0))[0]
    assert reference["applied"] == [False, False, True, False, False, True, False], (
        "only the microbatch closing a window may step")
    assert reference["optimizer_steps"] == 2, "the trailing partial window must be dropped"

    replicas = require_success(run_case("pp_pipeline_step", 2, options=options,
                                        hard_timeout=120.0))
    for payload in replicas:
        assert payload["applied"] == reference["applied"]
    joined = replicas[0]["parameters"] + replicas[1]["parameters"]
    same_weights(joined, reference["parameters"], "accumulated pipeline weights")


@pytest.mark.parametrize("mode", ["fsdp", "tp"])
def test_checkpoint_roundtrip_at_two_ranks(mode):
    """Save, rebuild from scratch, load, and compare -- through the real trainer.

    The persistence path had zero multi-rank coverage: every checkpoint test was
    single-process.  It is per-rank by design (each rank writes its own complete
    ``rank_state.pt``, and ``CheckpointManager.save`` issues no collective), so
    the things that can only break here are per-rank file writing and
    ``load_state_dict`` under a live process group.
    """
    replicas = require_success(run_case("checkpoint_roundtrip", 2, options={"mode": mode},
                                        hard_timeout=180.0))
    assert len(replicas) == 2
    for payload in replicas:
        assert payload["written"] == ["READY", "checksums.json", "metadata.json", "rank_state.pt"]
        assert payload["max_diff"] == 0.0, "restored parameters differ from the trained ones"
        assert payload["global_step"] == 1
        assert payload["optimizer_step"] == 1


def test_dcp_save_keeps_every_rank_shard():
    """A collective save must not let one rank's publish delete the others'.

    ``dcp.save`` is collective: each rank contributes a different shard and the
    coordinator writes ``.metadata``, all into one directory.  ``save_dcp`` used
    to stage in a per-process ``mkdtemp`` and then let EVERY rank ``_publish``
    over the shared target, so the last rank to finish replaced the directory the
    others had just written into.  Measured at world_size=2 before the fix: the
    target held only ``['__1_0.distcp']``, with rank 0's shard and ``.metadata``
    gone, and the load failed.

    Asserting the shard count is what makes this a regression test rather than a
    smoke test -- a one-shard directory still "saves successfully".
    """
    reference = require_success(run_case("checkpoint_dcp_roundtrip", 1,
                                         hard_timeout=180.0))[0]
    assert reference["shards"] == ["__0_0.distcp"]
    assert reference["max_diff"] == 0.0

    replicas = require_success(run_case("checkpoint_dcp_roundtrip", 2, hard_timeout=180.0))
    for payload in replicas:
        assert payload["shards"] == ["__0_0.distcp", "__1_0.distcp"], (
            "a shard or the .metadata file was destroyed by another rank's publish")
        assert payload["metadata"]["world"] == 2
        assert payload["max_diff"] == 0.0, "DCP round-trip did not restore the tensor"


def test_sharded_convert_and_load_across_ranks():
    """Dense -> sharded manifest -> per-rank load, over a live process group.

    ``CheckpointConverter.convert`` and ``ModelLoader.load_for_rank`` are the two
    entry points that make sharded loading possible, and both only ever had
    single-process tests.  Nothing in ``src/`` calls either -- ``Trainer`` writes
    a full replica on every rank -- so this is the only place their multi-rank
    behaviour is checked at all.

    The halves must be complementary and correct: the worker asserts each rank's
    tensor equals ``ref[rank*half:(rank+1)*half]``, which is what distinguishes
    real sharding from a load that quietly hands everyone the same thing.
    """
    for payload in require_success(run_case("sharded_convert_and_load", 2, hard_timeout=120.0)):
        assert payload["stats"]["tensors_loaded"] == 2
        assert payload["w_shape"] == [4, 4], "the partitioned tensor must arrive as a shard"
        assert payload["b_shape"] == [8], "an unpartitioned tensor stays full-size"
        assert len(payload["shard_files"]) == 2


def test_convert_records_the_axis_it_actually_sharded():
    """A bare ``world_size=N`` means tp=N, and the manifest must say so.

    This caught a real surprise: ``convert(source, dest, world_size=2)`` shards
    along TP, so loading with ``dp_rank=1`` fails.  Asserting the recorded
    ``logical_sharding`` keeps the default from being assumed rather than known.
    """
    for payload in require_success(run_case("sharded_convert_and_load", 2, hard_timeout=120.0)):
        assert payload["logical_sharding"] == {"dp": 1, "pp": 1, "tp": 2}


def test_dp_axis_replicates_rather_than_shards():
    """``dp_size`` fans copies out; it never splits a tensor.

    The converter chunks by ``effective_tp`` only.  So a dp-only conversion gives
    no storage saving -- faithful to what DP replicas are, but not what a reader
    of the argument name would guess, and worth pinning rather than discovering
    from a full disk.
    """
    for payload in require_success(run_case("sharded_convert_replicates_along_dp", 2,
                                            hard_timeout=120.0)):
        assert payload["shape"] == [8, 4], "dp must not split the tensor"
        assert payload["matches_whole_tensor"] is True


def test_loading_a_foreign_coordinate_is_refused():
    """A rank nothing addresses must raise, not silently keep its init.

    The failure mode this guards is quiet: the model keeps random weights while
    the load reports success.  The worker first checks that the coordinate it
    asks for is genuinely absent from the manifest's targets -- on rank 0,
    ``dp_rank=0`` is the same tuple as ``tp_rank=0`` and would legitimately
    match, so a carelessly chosen "foreign" coordinate proves nothing.
    """
    for payload in require_success(run_case("sharded_load_rejects_a_foreign_coordinate", 2,
                                            hard_timeout=120.0)):
        assert payload["refused"] is True
        assert "would keep its random initialisation" in payload["message"]
        assert [0, 0, 1] not in payload["targets"]


def test_sharded_checkpoint_roundtrip_at_two_ranks():
    """The opt-in DCP path: ``save_sharded`` / ``load_sharded`` over two ranks.

    AdamW rather than SGD in the worker, because plain SGD has no state and the
    optimizer assertions below would pass while proving nothing.  The shard count
    matters too: ``get_model_state_dict`` returns full-shaped tensors and DCP
    splits them for storage, so "one file" would mean no sharding happened.

    The LR schedule is checked here as well as single-process: the schedule's
    position used to be dropped by this format entirely, and the optimizer's own
    ``param_groups`` carry the current lr, so the loss of it is invisible until a
    scheduler that computes an lr from ``last_epoch`` steps again.
    """
    for payload in require_success(run_case("sharded_checkpoint_roundtrip", 2,
                                            hard_timeout=180.0)):
        assert payload["written"] == ["READY", "dcp", "metadata.json"]
        assert payload["shards"] == ["__0_0.distcp", "__1_0.distcp"], "not sharded"
        assert payload["format"] == "dcp-sharded"
        assert payload["max_diff"] == 0.0, "restored parameters differ"
        assert payload["global_step"] == 3 and payload["optimizer_step"] == 3
        assert payload["optimizer_state_entries"] > 0, (
            "the optimizer state did not survive; the checkpoint is not resumable")
        assert payload["restored_epoch"] == payload["saved_epoch"] == 3, (
            f"the schedule position did not survive: saved at "
            f"{payload['saved_epoch']}, restored at {payload['restored_epoch']}")
        # The step after the resume has to land on the same lr the ORIGINAL run
        # produced on the same step -- that is the value training consumes, and
        # the one a lost schedule position changes.
        assert payload["lr_after_step"] == payload["lr_reference"], (
            f"the schedule is off after a resume: {payload['lr_after_step']} "
            f"against the original run's {payload['lr_reference']}")


# Every axis combination this framework accepts at the world sizes a single
# machine can host.  pp+tp is absent on purpose -- see the note in the test.
SHARDED_AXIS_MATRIX = [
    (2, {"dp": 2}),
    (2, {"pp": 2}),
    (2, {"tp": 2}),
    (2, {"tp": 2, "sp": "megatron"}),
    (4, {"dp": 4}),
    (4, {"pp": 4}),
    (4, {"tp": 4}),
    (4, {"dp": 2, "tp": 2}),
    (4, {"dp": 2, "pp": 2}),
    (4, {"tp": 4, "sp": "megatron"}),
    (4, {"dp": 2, "tp": 2, "sp": "megatron"}),
    (4, {"tp": 2, "pp": 2}),
    (8, {"dp": 2, "tp": 2, "pp": 2}),
    (8, {"dp": 2, "tp": 2, "pp": 2, "sp": "megatron"}),
]


@pytest.mark.parametrize(("world", "options"), SHARDED_AXIS_MATRIX,
                         ids=[f"w{w}-" + "-".join(f"{k}{v}" for k, v in sorted(o.items()))
                              for w, o in SHARDED_AXIS_MATRIX])
def test_sharded_checkpoint_round_trips_axis_combinations(world, options):
    """Which parallel strategies the sharded checkpoint actually supports.

    This is the measured answer, not a reading of the code: for each axis
    combination the framework accepts, train a couple of steps, save
    collectively, rebuild from scratch, load, and compare -- over real
    processes.  Every entry below restores to max|dW| = 0.0e+00.

    ``per_rank_values_differ`` keeps the DP results from passing vacuously: with
    identical replicas a load that handed one rank's tensor to everybody would
    look exactly like a correct one.

    **pp+tp is missing because it does not work**, and the cause is upstream of
    checkpointing: ``split_sequential`` wraps each stage's children in its own
    ``nn.Sequential``, so every stage's module names become ``0, 1, ...``.  The
    TP plan matches declared suffixes (``q_proj``, ``o_proj``), finds none, and
    refuses -- measured as ``TPConfigurationError: the plan names no module to
    shard``.  Before the plan became name-based that combination silently ran
    with no tensor parallelism at all, so this is a loud version of a long-
    standing gap rather than a new one.  The same re-indexing is why PP stages
    share checkpoint keys, which is what ``checkpointing._stage_prefix`` works
    around.
    """
    payloads = require_success(run_case("sharded_roundtrip_axes", world, options=options,
                                        hard_timeout=300.0))
    assert len(payloads) == world
    for payload in payloads:
        assert payload["ran"] is True, payload.get("error")
        assert payload["max_diff"] == 0.0, (
            f"{options} round trip changed the weights")
        assert payload["steps"] == 2
        assert payload["optimizer_state_entries"] > 0, (
            "the optimizer state did not survive; the checkpoint is not resumable")
    if any(value != 1 for value in options.values() if isinstance(value, int)):
        assert any(p["per_rank_values_differ"] for p in payloads), (
            "every rank holds identical weights, so this case cannot detect a load "
            "that hands one rank's tensor to everybody")
    if options.get("pp", 1) > 1:
        assert not any(p["keys_collide_across_ranks"] for p in payloads), (
            "PP stages share checkpoint key names, so the same key means different "
            "tensors on different ranks -- DCP would keep one and hand it to both, "
            "which is what split_sequential used to cause by re-indexing every "
            "stage's children from 0")


def test_every_rank_builds_the_same_model():
    """``Runtime`` must not offset the seed by rank, or the ranks train different models.

    The global RNG is where a caller draws model initialisation from, and every
    example in this repository -- and this class's own docstring -- creates the
    ``Runtime`` *before* the model.  Offsetting the seed by rank therefore made
    each rank build a different model: measured at world=2 as
    ``torch.initial_seed() == 1234`` on rank 0 and ``1235`` on rank 1, with the
    "same" model differing by max|dW| = 6.9e-01 before a single step.

    Tensor parallelism hid it (``distribute_tensor`` broadcasts from
    ``src_data_rank``, overwriting the difference) and FSDP hid it more
    thoroughly (its all-gather reconstructs a parameter out of both ranks, so the
    result is half-initialised from each).  Pipeline parallelism has neither,
    which is where it surfaced.

    Data sharding is the caller's job, so the offset bought nothing.
    """
    payloads = require_success(run_case("composition_train_matches_dense", 2,
                                        options={"shape": "1,1,2"}, hard_timeout=180.0))
    assert len({p["initial_seed"] for p in payloads}) == 1, (
        "ranks disagree about the global seed, so they will build different models")
    first = payloads[0]["before_parallelize"]
    for payload in payloads[1:]:
        assert payload["before_parallelize"] == first, (
            "the same model built on two ranks came out different")


@pytest.mark.parametrize(("world", "shape", "sp"), [
    (2, "2,1,1", "none"),
    (2, "1,2,1", "none"),
    (2, "1,1,2", "none"),
    (4, "1,2,2", "none"),
    (4, "1,2,2", "megatron"),
    (8, "2,2,2", "none"),
    (8, "2,2,2", "megatron"),
])
def test_training_matches_a_single_process_on_every_axis(world, shape, sp):
    """DP+TP+PP+SP together must TRAIN to the same weights as one process.

    The checkpoint cases prove a round trip is lossless; this proves the
    training itself is right.  Every rank gets identical data, so a DP reduction
    over identical replicas is the identity and the world=1 run of the same case
    is a valid reference.

    The pipeline stages hold disjoint parameters, so their reported values are
    concatenated in stage order before comparing -- which also checks that the
    stage split put the right layers on the right rank.
    """
    reference = require_success(run_case("composition_train_matches_dense", 1,
                                         options={"shape": "1,1,1"},
                                         hard_timeout=120.0))[0]
    payloads = require_success(run_case("composition_train_matches_dense", world,
                                        options={"shape": shape, "sp": sp},
                                        hard_timeout=300.0))
    joined: list = []
    for stage in sorted({p["stage"] for p in payloads}):
        joined.extend(next(p["parameters"] for p in payloads if p["stage"] == stage))
    assert len(joined) == len(reference["parameters"]), "the stages do not partition the model"
    worst = max(abs(a - b) for a, b in zip(joined, reference["parameters"]))
    assert worst < TOLERANCE, (
        f"dp,tp,pp={shape} sp={sp} trained to different weights than one process "
        f"(max|diff| = {worst:.3e})")


def test_sharded_checkpoint_reshards_to_a_single_process():
    """The sharded checkpoint must be readable at a different world size.

    This is what "真分片" is for, and the framework did not have it: the old
    documentation recorded "换 world size 只能拒绝，不能转换".  Because the saved
    tensors are globally described, DCP can hand the whole thing to a single
    process with **no process group at all** (``no_dist=True``) -- a simultaneous
    change of world size *and* of TP layout.

    ``shapes_match`` is the part a numeric comparison alone would miss: reading a
    TP checkpoint into a dense model has to come back at the GLOBAL shape (8, 8),
    not as one rank's (4, 8) slice.
    """
    payloads = require_success(run_case("sharded_reshard_to_one_process", 2,
                                        hard_timeout=180.0))
    reader = payloads[0]
    assert reader["loaded"] is True, "rank 0 did not perform the resharding load"
    assert reader["max_diff"] == 0.0, "the resharded weights differ from the trained ones"
    assert reader["shape"] == [8, 8], "the tensor did not come back at its global shape"
    assert reader["shapes_match"] is True, (
        "at least one tensor's shape differs between the dense and the sharded view")


@pytest.mark.parametrize(("world", "options"), [
    (2, {"dp": 2}),
    (2, {"tp": 2}),
    (2, {"pp": 2}),
    (4, {"tp": 2, "dp": 2}),
    (4, {"tp": 2, "pp": 2}),
    (4, {"dp": 2, "pp": 2}),
    (8, {"dp": 2, "tp": 2, "pp": 2}),
], ids=["dp2", "tp2", "pp2", "tp2dp2", "tp2pp2", "dp2pp2", "dp2tp2pp2"])
def test_pretrained_load_reads_only_this_ranks_shards(world, options):
    """Pretrained loading must not be "rank 0 reads everything and broadcasts".

    The weights arriving correctly is the easy half; the other half is *how*, and
    it is invisible in the result -- a rank that reads the whole model and ships
    it out produces identical parameters.  So this asserts the read volume, which
    is the actual requirement: with a checkpoint converted for this topology each
    rank reads its own slice, and the per-rank bytes fall as ``1/(dp*tp*pp)``.

    It fails loudly on a topology mismatch rather than guessing: a checkpoint
    sharded for a different layout still produces a tensor of *some* shape, so a
    silent redistribute would cut the wrong thing.  ``_assign`` reports instead.

    The total is slightly above the model size (1.06x at tp>1) because a row
    projection's bias is genuinely replicated -- both ranks must read it.
    """
    payloads = require_success(run_case("load_pretrained_per_rank", world, options=options,
                                        hard_timeout=300.0))
    assert len(payloads) == world
    dense = payloads[0]["dense_bytes"]
    for payload in payloads:
        assert payload["max_diff"] == 0.0, "the loaded weights differ from the reference"
        assert 0 < payload["bytes_read"] < dense, (
            f"rank {payload['rank']} read {payload['bytes_read']} bytes for a {dense}-byte "
            "model -- it read everything, so nothing was saved over a broadcast")
    total = sum(payload["bytes_read"] for payload in payloads)
    # This is the assertion that catches "rank 0 reads everything and broadcasts":
    # then every rank reads a whole model and the total is world_size * dense.
    # A correctly partitioned checkpoint costs about one model in total, plus the
    # tensors that really are replicated (a row projection's bias, which both
    # ranks must read) -- which is why the bound is 1.15x and not 1.0x.  Per-rank
    # reads are NOT asserted to be under dense/2 for the same reason: at tp=2 the
    # replicated bias pushes one rank's total just above half.
    assert total < dense * 1.15, (
        f"the ranks together read {total} bytes for a {dense}-byte model; a correctly "
        "partitioned checkpoint costs about one model, not one per rank")


def test_pipeline_stages_keep_their_module_names():
    """The fix that lets pp and tp compose, pinned directly.

    ``split_sequential`` used to wrap each stage's children in an
    ``nn.Sequential``, which renames them by position.  Two things broke, both
    silently: a name-matching TP plan found nothing on a stage (so pp+tp ran
    with tensor parallelism switched off, or was refused once the plan became
    name-based), and every stage's checkpoint keys started at ``"0"``, so stage
    0's ``0.weight`` and stage 1's ``0.weight`` were different tensors under one
    key.  The reindexing is gone, so a stage now reports the caller's own names.

    The checkpoint side of the same property is asserted by
    ``test_sharded_checkpoint_round_trips_axis_combinations``; this asserts the
    names, which is what makes both work.
    """
    payloads = require_success(run_case("sharded_roundtrip_axes", 4,
                                        options={"tp": 2, "pp": 2}, hard_timeout=180.0))
    names = [set(p["keys"]) for p in payloads]
    for keys in names:
        assert keys, "the stage reported no parameters at all"
        assert not any(key.split(".")[0].rstrip("0123456789") == "" for key in keys), (
            f"stage children are named by position again: {sorted(keys)}")
    assert names[0] != names[-1], (
        "both stages report the same parameter names, so the stages were "
        "re-indexed rather than keeping their original module names")
    for payload in payloads:
        assert payload["ran"] is True, payload.get("error")
        assert payload["max_diff"] == 0.0


def test_data_parallel_ranks_can_shard_the_dataset():
    """``train_dataloader`` must receive the DP group, not ``None``.

    It was passed ``None`` unconditionally, so a provider had no way to tell one
    data-parallel rank from another and every rank iterated the whole dataset in
    the same order.  Data parallelism then degrades to N replicas averaging
    identical gradients: it trains, it just spends N times the compute for one
    rank's batch -- and nothing reports it.
    """
    for payload in require_success(run_case("data_parallel_group_is_handed_over", 2,
                                            hard_timeout=120.0)):
        assert payload["group_size"] == 2, (
            f"rank {payload['rank']} was handed no usable DP group "
            f"(size {payload['group_size']}), so no rank can shard the dataset")
        # The index a provider shards by.  A group that exists but gives both
        # ranks the same index is no better than no group at all: every rank
        # still reads shard 0.
        assert sorted(payload["indices"].values()) == [0, 1], (
            f"the DP group does not distinguish the ranks: {payload['indices']}")
        # ...and that the group reaches the provider through the training loop,
        # which is the line that was passing None.  Checking the helper alone
        # would still pass with `fit` handing the provider None.
        assert payload["fit_handed_it"], (
            "Trainer.fit did not pass the DP group to train_dataloader")
        assert payload["evaluate_handed_it"], (
            "Trainer.evaluate did not pass the DP group to train_dataloader")


def test_profiler_capture_measures_real_collectives():
    """``Profiler.capture`` must report an overlap it actually observed.

    The metric was structurally 0.0 because nothing ever called
    ``record_async``/``record_wait``: the collectives worth measuring live inside
    DTensor and FSDP2 as autograd nodes, with no call site to hang a timer on.
    Reading them out of a ``torch.profiler`` trace is the only way in.

    Both halves are asserted, because either alone is satisfiable by a broken
    implementation: a profiler that finds nothing reports 0.0 and looks like "no
    overlap", and one that never runs reports 0.0 too.  So the count has to be
    positive AND the split has to be a split.
    """
    payloads = require_success(run_case("profiler_captures_collectives", 2,
                                        hard_timeout=180.0))
    assert len(payloads) == 2
    for payload in payloads:
        assert payload["collectives"] > 0, (
            f"rank {payload['rank']} captured no collectives, so the metric is the "
            "structural zero it used to be")
        assert payload["timeline_entries"] == payload["collectives"], (
            "every captured collective must reach the timeline")
        assert payload["exposed_seconds"] + payload["hidden_seconds"] > 0
        assert 0.0 <= payload["overlap_ratio"] <= 1.0
    # The ranks run the same three steps, so they must agree on how many
    # collectives there were; the times differ legitimately between ranks.
    assert len({payload["collectives"] for payload in payloads}) == 1, (
        f"ranks disagree on the collective count: "
        f"{[payload['collectives'] for payload in payloads]}")


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_every_pipeline_stage_receives_gradients(schedule):
    """An interior pipeline stage must train, not just pass gradients through.

    ``run_middle_stage`` backpropagated the downstream gradient through its own
    INPUT -- a tensor received from another process, whose graph holds none of
    this stage's parameters.  An interior stage therefore never accumulated a
    gradient on any parameter, while the loss stayed correct and
    ``backward_complete`` was True.

    Four stages, because the interior path only exists when the world has an
    interior: at pp=2 this is never exercised, and every two-stage run in the
    suite was green.

    Both schedules, because they reach the interior by different code:
    ``GPipeSchedule.run_middle_stage`` and ``OneFOneBSchedule._run`` each
    backpropagated the received activation instead of the produced output, and
    each had to be fixed separately.

    The in-process protocol simulator did not catch either.  Its assertions are
    about what gets SENT -- message order, stage labels, microbatch labels -- and
    the forwarded gradient is non-empty with the right shape, so a gradient that
    never passed through this stage's Jacobian looks the same to it as one that
    did.
    """
    payloads = require_success(run_case("interior_pipeline_stage_trains", 4,
                                        options={"schedule": schedule}, hard_timeout=180.0))
    assert len(payloads) == 4
    stages = sorted(payload["stage"] for payload in payloads)
    assert stages == [0, 1, 2, 3], f"expected four stages, got {stages}"
    assert {payload["schedule"] for payload in payloads} == {schedule}
    for payload in payloads:
        assert payload["parameters"] > 0
        assert payload["without_gradient"] == 0, (
            f"stage {payload['stage']} has {payload['without_gradient']} parameters with no "
            "gradient at all, so it is not training")
        assert payload["max_grad"] > 0.0, (
            f"stage {payload['stage']} has only zero gradients")
        assert payload["backward_complete"] is True
    # Only the last stage owns the loss; the others report zero by design.
    last = max(payloads, key=lambda item: item["stage"])
    assert last["loss"] > 0.0
