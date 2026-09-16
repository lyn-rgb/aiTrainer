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


def test_stage_assignment_is_derived_from_the_same_split_training_uses():
    """The mapping ``convert`` needs must come from the model, not from the caller.

    ``convert`` refused ``pp_size>1`` without a mapping, and the reason was
    sound -- a dense state dict does not say which layer belongs to which stage.
    But the framework *does* know: it performs that split at training time.  This
    derives the mapping by running the same ``split_sequential``, so the
    checkpoint and the training run cannot disagree about where a layer went.

    It also pins the names.  With the old position-renaming container every
    stage reported ``0.weight``, so a derived mapping would have had one entry
    per stage all called ``0.weight`` -- useless, and silently so.
    """
    import torch

    from aitrainer import stage_assignment

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.middle = torch.nn.ReLU()
            self.o_proj = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.o_proj(self.middle(self.q_proj(value)))

    mapping = stage_assignment(Block(), 2)
    assert mapping == {"q_proj.weight": 0, "q_proj.bias": 0,
                       "o_proj.weight": 1, "o_proj.bias": 1}, mapping
    assert set(mapping) == {name for name, _ in Block().named_parameters()}, (
        "every parameter must be assigned, or convert would write it to a default stage")


def test_convert_accepts_a_derived_stage_assignment(tmp_path):
    """End to end: derive the mapping, convert a per-stage sharded checkpoint."""
    import torch

    from aitrainer import CheckpointConverter, stage_assignment

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.middle = torch.nn.ReLU()
            self.o_proj = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.o_proj(self.middle(self.q_proj(value)))

    model = Block()
    torch.save(model.state_dict(), tmp_path / "dense.pt")
    manifest = CheckpointConverter().convert(
        tmp_path / "dense.pt", tmp_path / "sharded", pp_size=2,
        stage_assignment=stage_assignment(model, 2))
    assert manifest.logical_sharding["pp"] == 2
    per_stage: dict[int, set] = {}
    for name, shards in manifest.tensors.items():
        for shard in shards:
            per_stage.setdefault(shard.target_rank[0], set()).add(name)
    assert per_stage[0] == {"q_proj.weight", "q_proj.bias"}, per_stage
    assert per_stage[1] == {"o_proj.weight", "o_proj.bias"}, per_stage
