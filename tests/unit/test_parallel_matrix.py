"""Contract tests for the server-side FSDP/SP/TP/PP validation matrix."""

from scripts.parallel_matrix import _specs, _validate_entry


def test_parallel_matrix_contains_all_requested_combinations():
    names = [spec.name for spec in _specs(2)]
    assert names == ["fsdp", "fsdp_sp", "fsdp_sp_tp", "fsdp_sp_tp_pp"]


def test_combined_fsdp_entries_are_capability_validated():
    entries = {}
    for spec in _specs(2):
        entry = _validate_entry(spec)
        entries[entry["name"]] = entry
    for name in ("fsdp", "fsdp_sp", "fsdp_sp_tp", "fsdp_sp_tp_pp"):
        assert entries[name]["status"] == "validated"
        assert entries[name]["expected_status"] == "supported"
