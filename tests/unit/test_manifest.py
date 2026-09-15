from aitrainer.checkpoint import Manifest, ShardSpec, load_manifest, write_manifest


def test_manifest_round_trip(tmp_path):
    manifest = Manifest(tensors={"weight": (ShardSpec("weight", (4, 2), "float32", "replicated"),)})
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert load_manifest(path) == manifest
