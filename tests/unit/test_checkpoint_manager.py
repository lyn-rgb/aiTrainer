import pytest

from aitrainer.checkpoint import CheckpointManager, IncompleteCheckpointError, Manifest, ShardSpec
from aitrainer.checkpoint.format import load_manifest, write_manifest


def test_manifest_atomic_round_trip_and_validation(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(tensors={"weight": (ShardSpec("weight", (2, 2), "float32", "replicated"),)})
    write_manifest(manifest, path)
    assert load_manifest(path) == manifest


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IncompleteCheckpointError):
        CheckpointManager()._verify(path)
