import pytest

from aitrainer.mesh import DeviceMeshManager
from aitrainer.topology import RankCoordinate, TopologyError, make_rank_mapping


def test_mapping_coordinates_and_groups_are_explicit():
    mapping = make_rank_mapping(4, dp_size=2, tp_size=1, pp_size=2, global_ranks=(2, 3, 0, 1))
    assert mapping.rank(RankCoordinate(pp=0, dp=0, tp=0)) == 2
    assert mapping.coordinate(0) == RankCoordinate(pp=1, dp=0, tp=0)
    assert mapping.group_ranks("dp", RankCoordinate(pp=0, dp=0, tp=0)) == (2, 3)


def test_mapping_rejects_duplicate_or_mismatched_ranks():
    with pytest.raises(TopologyError):
        make_rank_mapping(2, dp_size=2, global_ranks=(0, 0))
    with pytest.raises(TopologyError):
        make_rank_mapping(3, dp_size=2)


def test_mesh_metadata_works_without_initialized_process_group():
    mesh = DeviceMeshManager(world_size=2, dp_size=2)
    assert mesh.coordinate.dp == 0
    assert mesh.ranks("dp") == (0, 1)
