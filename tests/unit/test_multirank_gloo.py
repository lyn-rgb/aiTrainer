"""Real multi-process checks on a live Gloo process group.

Every other "distributed" test in this repository runs in a single process and
checks shapes at ``world_size == 1``.  These launch one OS process per rank, so
the items ``docs/代码审计报告.md`` records as *fixed but not verifiable without a
real multi-rank job* are exercised for real:

* §4.10 a -- canonical ``new_group`` ordering, with the removed ordering kept as
  a negative control that must still deadlock;
* §4.10 b -- gathered column-parallel bias against a dense reference;
* §4.10 c -- sequence-parallel LayerNorm parameter-gradient reduction;
* §4.10 d -- the variable-length Ulysses refusal;
* §4.1 -- ``no_sync`` accumulation across replicas.

``torchrun`` cannot start here (its rendezvous resolves the hostname through
mDNS and dies with ``gai error: 8``), but that is a property of ``torchrun``,
not of multi-rank: setting the rendezvous variables by hand works.  Hence
``tests/multirank/harness.py``.  A module-level ``importorskip`` would hide this
whole file, so each test asserts on a ``RESULT`` payload rather than on an exit
code alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.multirank.harness import require_success, run_case

TOLERANCE = 1e-6


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
                       timeout_seconds=3.0, hard_timeout=10.0)
    assert not all(result.returncode == 0 for result in results), (
        "the removed ordering completed cleanly, so the canonical ordering fix "
        "is not what makes group creation terminate")
    combined = " ".join((result.error or "") + result.stderr for result in results)
    assert "wait timeout" in combined or "TimeoutExpired" in combined, (
        f"expected a store-barrier timeout from the legacy ordering, got:\n{combined[-2000:]}")


def test_column_parallel_gather_output_bias_matches_dense():
    """§4.10 b: the shard-width bias must be added before gathering."""
    for payload in require_success(run_case("column_parallel_bias", 2, hard_timeout=60.0)):
        assert payload["forward_error"] < TOLERANCE
        assert payload["fused_error"] < TOLERANCE, "skip_bias_add returned a shard-width bias"
        assert payload["input_grad_error"] < TOLERANCE


def test_sequence_parallel_layernorm_gradients_match_dense():
    """§4.10 c: replicated norm parameters need the TP-reduced gradient."""
    for payload in require_success(run_case("sp_layernorm_grad_reduction", 2, hard_timeout=60.0)):
        assert payload["forward_error"] < TOLERANCE
        assert payload["weight_grad_error"] < TOLERANCE
        assert payload["bias_grad_error"] < TOLERANCE


def test_sequence_parallel_layernorm_reduction_is_load_bearing():
    """Without the reduction each rank's shard gives a different gradient."""
    payloads = require_success(run_case("sp_layernorm_without_reduction_diverges", 2,
                                        hard_timeout=60.0))
    left, right = payloads[0]["weight_grad"], payloads[1]["weight_grad"]
    worst = max(abs(a - b) for a, b in zip(left, right))
    assert worst > 0.5, (
        "the unreduced shard gradients agree, so this case does not demonstrate "
        f"that the TP-group reduction changes anything (max difference {worst})")


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
    results = run_case("ulysses_attention_equivalence", 2, hard_timeout=60.0)
    assert all(result.returncode != 0 for result in results)
    assert all("alltoall" in (result.error or "") for result in results), (
        "expected an unsupported-alltoall failure from Gloo")


def test_fsdp_initialises_on_a_cpu_only_host():
    """FSDP must accept a resolved CPU device instead of inferring one.

    FSDP's own device inference resolves the custom 'mps' backend on a CPU-only
    macOS host and raises ``Custom backend 'mps' not implement
    torch.mps.current_device``.  ``wrap_fsdp`` therefore has to forward the
    device the runtime resolved rather than forwarding CUDA only.
    """
    for payload in require_success(run_case("fsdp_cpu_wrap", 2, hard_timeout=60.0)):
        assert payload["wrapped"] == "FSDPWrapper"


def test_trainer_fit_over_fsdp_matches_single_process():
    """§4.1 end to end: the real ``Trainer``, real FSDP, real DP-group reduction.

    Both ranks are fed identical data, so the reduced gradient equals the
    single-process gradient and the parameters must match a ``world_size == 1``
    run exactly.  Two-process agreement alone would not catch a *consistently*
    wrong reduction or a mis-scaled update; the reference does.

    The batch count is not a multiple of ``grad_accumulation_steps``, so a
    trailing partial window exists and must be dropped rather than flushed --
    re-adding the flush makes the multi-rank run fail outright.  This case also
    exercises ``full_state_dict_context``, whose ``offload_to_cpu=True`` used to
    segfault the process when the parameters were already on CPU.
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
    assert joined == reference["parameters"], "pipeline parameters diverged from single-process"
    assert replicas[1]["losses"] == reference["losses"], "last-stage loss diverged"
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
    assert joined == reference["parameters"], "accumulated pipeline weights diverged"
