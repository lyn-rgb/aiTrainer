import pytest

from aitrainer import TransformerTPPlan


def test_plan_roles_come_from_declared_suffixes_only():
    plan = TransformerTPPlan()
    assert plan.role("layers.0.q_proj") == "column"
    assert plan.role("layers.0.o_proj") == "row"
    assert plan.role("layers.0.mystery_proj") is None, (
        "an undeclared name must stay unsharded rather than be guessed at")
    assert plan.role("lm_head") is None


def test_styles_shard_leaves_and_never_restructure_the_tree():
    """The plan names FQNs; the modules themselves are left alone.

    This is the property the checkpointing migration depends on: because
    ``parallelize_module`` replaces a submodule's *parameters* with DTensors
    rather than swapping the submodule, ``state_dict`` keys are unchanged.
    """
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.o_proj = torch.nn.Linear(8, 8)
            self.lm_head = torch.nn.Linear(8, 4)

    styles = TransformerTPPlan().styles(Block())
    assert sorted(styles) == ["o_proj", "q_proj"], "an unnamed Linear must not be sharded"
    kinds = {name: type(style).__name__ for name, style in styles.items()}
    assert kinds == {"q_proj": "ColwiseParallel", "o_proj": "RowwiseParallel"}


def test_dimension_validation_names_the_offending_projection():
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(6, 6)

    with pytest.raises(ValueError, match="q_proj.out_features=6"):
        TransformerTPPlan().validate_dimensions(Block(), tp_size=4)


def test_sequence_parallel_styles_cover_the_norms_only():
    torch = pytest.importorskip("torch")
    from aitrainer.parallel.tp import sequence_parallel_styles

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)
            self.inner = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 8))

    assert sorted(sequence_parallel_styles(Block())) == ["inner.0", "norm"]
