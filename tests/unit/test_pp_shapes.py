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


def _llama_shaped(torch, layers: int = 4):
    """A model shaped the way HuggingFace ships one, layers one level down.

    ``LlamaForCausalLM``'s direct children are the trunk and the head; the four
    transformer layers live inside ``model.layers``.  That nesting is the whole
    point of these two tests.
    """

    class LlamaShaped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 8)
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Linear(8, 8) for _ in range(layers)])
            self.model.norm = torch.nn.LayerNorm(8)
            self.lm_head = torch.nn.Linear(8, 32)

        def forward(self, ids):
            hidden = self.model.embed_tokens(ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return self.lm_head(self.model.norm(hidden))

    return LlamaShaped()


def _llama_execution_order(model):
    """What ``ModelAdapter.execution_order`` would return for a HF model."""
    return [("embed_tokens", model.model.embed_tokens),
            *[(f"layer_{index}", layer) for index, layer in enumerate(model.model.layers)],
            ("norm", model.model.norm),
            ("lm_head", model.lm_head)]


def test_pipeline_split_refuses_a_model_it_could_never_run():
    """A split that cannot execute must be refused, not returned.

    Measured on this module at ``pp_size=2`` before the check existed: the split
    came back without complaint as ``['model']`` and ``['lm_head']``, putting the
    entire transformer on stage 0.  Nothing surfaced until stage 0 was first
    called, and then as

        NotImplementedError: Module [Module] is missing the required "forward"
        function

    which names neither the pipeline split nor the module responsible.  ``nn.ModuleList``
    -- how every HuggingFace model holds its layers -- has no forward either.

    The refusal does not rest on "this looks unbalanced": a ``PipelineSequence``
    runs each child by calling it, so no assignment of these children to stages
    can run, which is a property of the module rather than of the policy.
    """
    torch = pytest.importorskip("torch")
    from aitrainer.parallel.pp_shapes import split_sequential

    with pytest.raises(PipelineShapeError, match="defines no forward"):
        split_sequential(_llama_shaped(torch), 2)

    # Counterfactual: the same model with the same pp_size splits as soon as the
    # caller says what the units are, so the refusal is about the missing
    # description, not about nested models being unsupported.
    stages, plans = split_sequential(_llama_shaped(torch), 2,
                                     execution_order=_llama_execution_order)
    assert [stage.layer_names for stage in stages] == [
        ["embed_tokens", "layer_0", "layer_1", "layer_2"],
        ["layer_3", "norm", "lm_head"],
    ]
    assert [plan.stage for plan in plans] == [0, 1]


def test_execution_order_splits_a_nested_model_into_runnable_stages():
    """The split has to land on the layers, and each stage has to run.

    Both halves matter.  The parameter counts are what show the split landed
    where it was asked to -- the layers are what has to be divided, and the glue
    has to end up somewhere runnable.  And the stages are then called, because a
    split that names the right children in the wrong order produces a stage that
    accepts the input and returns the wrong shape.
    """
    torch = pytest.importorskip("torch")
    from aitrainer.parallel.pp_shapes import split_sequential

    torch.manual_seed(0)
    model = _llama_shaped(torch)
    stages, _ = split_sequential(model, 2, execution_order=_llama_execution_order)

    # The split does not copy: ``PipelineSequence`` registers the same child
    # objects the model holds, which is why the stages below can be compared
    # against the model itself rather than against a re-loaded copy.
    assert stages[0].layer_1 is model.model.layers[1]

    # Stage 0 owns the embedding and three of the four layers; stage 1 owns the
    # last layer and the head.  Naming the right children in the wrong order
    # would still produce runnable stages, so the parameter counts are what show
    # the layer stack was actually divided rather than gathered onto one rank.
    assert sum(parameter.numel() for parameter in stages[0].parameters()) == 472
    assert sum(parameter.numel() for parameter in stages[1].parameters()) == 376

    first = stages[0](torch.randint(0, 32, (2, 4)))
    assert tuple(first.shape) == (2, 4, 8), "stage 0 must emit the hidden width"
    assert tuple(stages[1](first).shape) == (2, 4, 32), "stage 1 must emit the vocabulary"

    # Chaining the stages reproduces the unsplit model exactly, which is the
    # property a pipeline split exists to preserve.
    ids = torch.randint(0, 32, (2, 4))
    with torch.no_grad():
        assert torch.equal(stages[1](stages[0](ids)), model(ids))


def test_execution_order_rejects_a_unit_name_a_stage_cannot_hold():
    """``add_module`` rejects ``'.'``, so the split must say so first.

    The natural mistake is to hand back the model's own qualified name
    (``"model.layers.0"``).  Without this the caller gets ``KeyError: module
    name can't contain "."`` from inside ``nn.Module``, which does not mention
    ``execution_order`` at all.
    """
    torch = pytest.importorskip("torch")
    from aitrainer.parallel.pp_shapes import split_sequential

    def dotted(model):
        return [("model.layers.0", model.model.layers[0]), ("lm_head", model.lm_head)]

    with pytest.raises(PipelineShapeError, match="must not contain"):
        split_sequential(_llama_shaped(torch), 2, execution_order=dotted)
