import pytest

from aitrainer.parallel.pp_shapes import PipelineShapeError, plan_stages, split_microbatches


def test_stage_plan_is_non_empty_and_balanced_by_layers():
    layers = [object() for _ in range(4)]
    plans = plan_stages(layers, 2)
    assert [(item.start, item.stop) for item in plans] == [(0, 2), (2, 4)]


def test_stage_plan_rejects_empty_stage():
    with pytest.raises(PipelineShapeError):
        plan_stages([object()], 2)


def test_microbatch_split_preserves_mapping_contract():
    batch = {"x": [1, 2, 3, 4], "labels": [5, 6, 7, 8]}
    # Non-tensor values are replicated deliberately; tensors are split by dim 0.
    assert len(split_microbatches(batch, 2)) == 2
