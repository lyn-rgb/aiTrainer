import pytest

torch = pytest.importorskip("torch")

from aitrainer.checkpoint import CheckpointConverter
from aitrainer.checkpoint.converter import ConversionError
from aitrainer.checkpoint.format import load_manifest


def test_dense_checkpoint_converter_writes_rank_shards(tmp_path):
    source = tmp_path / "dense.pt"
    torch.save({"model": {"weight": torch.arange(8).reshape(4, 2), "bias": torch.ones(4)}}, source)
    destination = tmp_path / "sharded"
    manifest = CheckpointConverter(partition_dims={"weight": 0}).convert(source, destination, world_size=2)
    assert manifest.world_size_at_save == 2
    assert tuple(manifest.tensors["weight"][0].local_shape) == (2, 2)
    assert load_manifest(destination / "manifest.json") == manifest


def test_dense_converter_refuses_pp_without_a_stage_assignment(tmp_path):
    """PP cannot be derived from a dense state dict.

    The converter previously advertised ``logical_sharding["pp"] = pp_size``
    while handing every pp_rank the SAME chunk, so each stage file held the whole
    model; loading one into a stage-local model then failed.
    """
    source = tmp_path / "dense.pt"
    torch.save({"model": {"weight": torch.arange(16).reshape(8, 2)}}, source)
    with pytest.raises(ConversionError, match="stage_assignment"):
        CheckpointConverter(partition_dims={"weight": 0}).convert(
            source, tmp_path / "combined", world_size=8, dp_size=2, tp_size=2, pp_size=2)


def test_dense_converter_partitions_pp_when_the_assignment_is_given(tmp_path):
    source = tmp_path / "dense.pt"
    torch.save({"model": {"weight": torch.arange(16).reshape(8, 2)}}, source)
    destination = tmp_path / "combined"
    manifest = CheckpointConverter(partition_dims={"weight": 0}).convert(
        source, destination, world_size=8, dp_size=2, tp_size=2, pp_size=2,
        stage_assignment={"weight": 1},
    )
    assert manifest.logical_sharding == {"dp": 2, "pp": 2, "tp": 2}
    # the tensor belongs to stage 1 only, sharded across tp and replicated across dp
    assert {item.target_rank for item in manifest.tensors["weight"]} == {(1, tp, dp)
                                                                        for tp in range(2) for dp in range(2)}
    # global_rank(pp,dp,tp) = (pp*dp_size + dp)*tp_size + tp -> pp=1 starts at rank 4
    stage0 = torch.load(destination / "rank_00000.pt", weights_only=False)
    stage1 = torch.load(destination / "rank_00004.pt", weights_only=False)
    assert "weight" not in stage0 and "weight" in stage1
