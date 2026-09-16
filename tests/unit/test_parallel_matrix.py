"""Contract tests for the server-side FSDP/SP/TP/PP validation matrix."""

from scripts.parallel_matrix import _specs, _validate_entry

# The combinations the matrix is supposed to cover.  There used to be a fourth,
# `fsdp_sp_tp`, described as "the Ulysses/TP path" -- it differed from `fsdp_sp`
# only in `sp_backend`, a field that selects nothing (one SP implementation), so
# the two ran the same configuration and produced bit-identical losses.  It was
# removed rather than renamed: it added no coverage, and its name invited the
# reading that Ulysses was exercised.  The list is pinned here so an entry can
# only disappear deliberately.
EXPECTED_ENTRIES = ["fsdp", "fsdp_sp", "fsdp_sp_tp_pp"]


def test_parallel_matrix_contains_all_requested_combinations():
    names = [spec.name for spec in _specs(2)]
    assert names == EXPECTED_ENTRIES


def test_combined_fsdp_entries_are_capability_validated():
    entries = {}
    for spec in _specs(2):
        entry = _validate_entry(spec)
        entries[entry["name"]] = entry
    for name in EXPECTED_ENTRIES:
        assert entries[name]["status"] == "validated"
        assert entries[name]["expected_status"] == "supported"


def test_no_two_entries_run_the_same_configuration():
    """Two entries that differ only in an ignored field are one entry.

    That is exactly how `fsdp_sp_tp` survived: it looked like a distinct
    combination, and the only thing setting it apart was `sp_backend`, whose
    `ulysses` value was reinterpreted as `megatron` by `parallelize`.  Comparing
    the resolved topology is what makes that visible without running the matrix.
    """
    seen = {}
    for spec in _specs(4):
        resolved = (spec.dp_size, spec.tp_size, spec.pp_size,
                    spec.sp_backend, spec.pp_schedule, spec.num_microbatches)
        assert resolved not in seen, (
            f"{spec.name} runs the same configuration as {seen[resolved]}: {resolved}")
        seen[resolved] = spec.name
