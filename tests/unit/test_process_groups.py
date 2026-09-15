"""`ProcessGroups.create` must issue identical `new_group` calls on every rank.

This is the fix for the >= 3-rank startup deadlock: creating only a rank's own
groups made torch's per-process call counter collide in the store, so one index
carried groups of DIFFERENT sizes, no rank observed the expected barrier count,
and every participant blocked for the full 30-minute gloo timeout.

`torch.distributed` is stubbed so the call sequence is observable without a
rendezvous: the first argument of every `new_group` call is recorded per rank.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch.distributed as dist

from aitrainer.parallel.groups import ProcessGroups, group_creation_plan
from aitrainer.topology import make_rank_mapping

MESHES = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 2, 2), (4, 2, 2)]


def _create_for(rank: int, mapping):
    """Run ProcessGroups.create as `rank`, returning (groups, recorded calls)."""
    calls: list[tuple[int, ...]] = []
    original = (dist.new_group, dist.is_initialized, dist.get_world_size, dist.get_rank)

    def new_group(ranks, **_kwargs):
        calls.append(tuple(ranks))
        return ("handle", tuple(ranks))

    dist.new_group = new_group
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda group=None: mapping.world_size
    dist.get_rank = lambda group=None: rank
    try:
        return ProcessGroups.create(mapping), calls
    finally:
        (dist.new_group, dist.is_initialized, dist.get_world_size, dist.get_rank) = original


@pytest.mark.parametrize("dp,tp,pp", MESHES)
def test_every_rank_issues_an_identical_call_sequence(dp, tp, pp):
    """The invariant `dist.new_group` requires: same order, same rank lists, every rank."""
    world = dp * tp * pp
    mapping = make_rank_mapping(world, pp_size=pp, dp_size=dp, tp_size=tp)
    sequences = []
    for rank in range(world):
        groups, calls = _create_for(rank, mapping)
        sequences.append(calls)

        # the handle retained per axis must be the group that CONTAINS this rank
        for axis, handle in (("pp", groups.pp_group), ("dp", groups.dp_group),
                             ("tp", groups.tp_group)):
            size = {"pp": pp, "dp": dp, "tp": tp}[axis]
            if world == 1:
                # no rendezvous and no groups needed for a single rank
                assert handle is None
                continue
            assert handle is not None, f"rank {rank} has no {axis} group"
            assert handle[1].count(rank) == 1, f"rank {rank} not in its own {axis} group"
            assert len(handle[1]) == size, f"{axis} group has {len(handle[1])} members, want {size}"

    first = sequences[0]
    if world == 1:
        assert first == []
        return
    assert first, "no groups were created"
    for rank, calls in enumerate(sequences):
        assert calls == first, f"rank {rank} issued a different sequence:\n{rank}: {calls}\n0: {first}"


@pytest.mark.parametrize("dp,tp,pp", MESHES)
def test_create_follows_the_canonical_plan_exactly(dp, tp, pp):
    """create() must issue exactly the plan, and the plan must cover each axis once."""
    world = dp * tp * pp
    mapping = make_rank_mapping(world, pp_size=pp, dp_size=dp, tp_size=tp)
    _groups, calls = _create_for(0, mapping)
    if world == 1:
        assert calls == []                       # single rank needs no process group
        return

    plan = group_creation_plan(mapping)
    assert calls == [ranks for _axis, ranks in plan]

    # Count per AXIS from the labelled plan.  Two notes on why this is per-axis:
    # a bare length count conflates axes that share a size (tp_size == pp_size == 1
    # both give length-1 groups), and identical rank lists on DIFFERENT axes are
    # legitimate -- dist.new_group([0]) called twice yields two distinct groups.
    for axis, size in (("pp", pp), ("dp", dp), ("tp", tp)):
        on_axis = [ranks for name, ranks in plan if name == axis]
        assert len(on_axis) == world // size, f"{axis}: {len(on_axis)} groups, want {world // size}"
        for rank in range(world):
            assert sum(rank in ranks for ranks in on_axis) == 1, f"{axis}: rank {rank} not in exactly one group"


def test_the_four_rank_two_by_two_sequence_is_the_verified_canonical_one():
    """Pin the exact sequence the A/B test showed working (4/4) versus hanging (0/4)."""
    _groups, calls = _create_for(0, make_rank_mapping(4, pp_size=1, dp_size=2, tp_size=2))
    assert calls == [(0,), (1,), (2,), (3,), (0, 2), (1, 3), (0, 1), (2, 3)]


def test_single_rank_mesh_creates_no_groups():
    """A single rank has nothing to shard over, so it skips the rendezvous entirely."""
    groups, calls = _create_for(0, make_rank_mapping(1, pp_size=1, dp_size=1, tp_size=1))
    assert calls == []
    assert groups.pp_group is None and groups.dp_group is None and groups.tp_group is None
