import pytest

from aitrainer import ConfigPreset, ConfigurationError, FrameworkConfig, dry_run


def test_default_config_is_valid_and_serializable():
    config = ConfigPreset.single_gpu()
    config.validate()
    assert FrameworkConfig.from_dict(config.to_dict()) == config


def test_config_is_immutable():
    config = FrameworkConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.seed = 1


@pytest.mark.parametrize("kwargs", [
    {"grad_accumulation_steps": 0},
    {"parallel": {"pp_size": 2}},
    {"compile": {"enabled": True}},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ConfigurationError):
        FrameworkConfig.from_dict(kwargs)


def test_dry_run_reports_capability_boundary():
    result = dry_run()
    assert result["status"] == "ok"
    assert any(item["name"] == "fsdp_full_shard" and item["status"] == "stable" for item in result["capabilities"])


def test_fsdp_preset_enables_only_data_parallel_path():
    config = ConfigPreset.fsdp()
    assert config.fsdp.enabled
    config.replace(parallel=config.parallel).validate()


def test_nested_configs_are_built_generically():
    """Nesting is resolved from the annotations, not from hand-written branches.

    ``from_dict`` used to carry one special case for ``fsdp.mixed_precision``,
    so a second nested config would have needed a second branch and forgetting
    one produced a bare TypeError from the dataclass constructor.
    """
    config = FrameworkConfig.from_dict(
        {"fsdp": {"enabled": True, "mixed_precision": {"enabled": True, "dtype": "float16"}}})
    assert config.fsdp.mixed_precision.enabled is True
    assert config.fsdp.mixed_precision.dtype == "float16"

    # A typo two levels down must still surface as a ConfigurationError, not as
    # the dataclass's own TypeError.
    with pytest.raises(ConfigurationError):
        FrameworkConfig.from_dict({"fsdp": {"mixed_precision": {"enabledd": True}}})


def test_trainer_construction_validates_the_config_once():
    """Constructing a Trainer must validate the config exactly once.

    ``Trainer.__init__`` called ``config.validate`` and then called
    ``validate_capabilities``, which calls it again -- so every construction
    checked the same config twice, and ``from_model`` (which also routes through
    ``parallelize``) made it three times.  Counting the calls is the only way to
    notice the redundancy creeping back.
    """
    torch = pytest.importorskip("torch")
    from aitrainer import FrameworkConfig, Trainer

    calls = []
    original = FrameworkConfig.validate

    def counting_validate(self, world_size=1):
        calls.append(world_size)
        return original(self, world_size=world_size)

    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    FrameworkConfig.validate = counting_validate
    try:
        Trainer(model, optimizer, config=FrameworkConfig(device="cpu", seed=5))
    finally:
        FrameworkConfig.validate = original
    assert len(calls) == 1, f"config validated {len(calls)} times during construction"


def test_fsdp_requires_a_process_group():
    """FSDP2 cannot invent a device mesh, and torch's own failure misleads.

    ``fully_shard`` with no mesh performs an env:// rendezvous, so a
    single-process caller that forgot to initialize a group got
    ``environment variable RANK expected, but not set`` -- an error that names
    neither FSDP nor the missing process group.  ``wrap_fsdp`` builds the mesh
    itself and therefore gets to say what is actually wrong.

    This replaced a test for ``forward_prefetch``/``execution_trace_complete``:
    both fields are gone with FSDP1, since neither had an FSDP2 counterpart.
    """
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    from aitrainer.parallel.fsdp import FSDPConfigurationError, wrap_fsdp

    if dist.is_initialized():
        pytest.skip("the process group is already initialized, so the guard cannot fire")

    class _Runtime:
        world_size = 1

        class state:
            device = "cpu"

    with pytest.raises(FSDPConfigurationError, match="process group"):
        wrap_fsdp(torch.nn.Linear(3, 2), runtime=_Runtime(), mesh=None, config=None)


def test_every_config_field_is_read_somewhere():
    """A config field with no reader is a lie the whole system keeps telling.

    It validates, it serialises into `to_dict()`, it shows up in the CLI's
    `dry-run` output and in checkpoint metadata -- so it looks live from every
    angle a user can see, while changing it alters nothing.  That is the worst
    shape of defect this project has: silent, and self-concealing.

    It has happened at least seven times: `PrecisionConfig.param_dtype`,
    `PrecisionConfig.reduce_dtype`, `FSDPConfig.sharding` / `limit_all_gathers` /
    `forward_prefetch` / `execution_trace_complete`, `CompileConfig.mode` /
    `fullgraph` / `dynamic`, and `OverlapConfig.enable_parameter_prefetch`.

    `reduce_dtype` is the one that shows why reading the code is not enough.  It
    was genuinely wired, to `RowParallelLinear`'s reduction, and then the DTensor
    migration deleted that layer and replaced it with a `RowwiseParallel` style
    (which takes no dtype).  The field, its validation, the two presets that set
    it and a docstring asserting "threaded into the TP row projections" all
    survived the code that did the work.  `test_precision` could not see any of
    it: nothing fails when a field stops being read.

    A reader is a `.field` access or a `getattr(obj, "field")` outside
    `schema.py`.  Validation does NOT count -- a field's own type check is
    precisely what a dead field has left.
    """
    import ast
    from dataclasses import fields, is_dataclass
    from pathlib import Path

    from aitrainer.config import schema as schema_module

    def leaf_names(obj, prefix=""):
        if is_dataclass(obj):
            names = []
            for field_ in fields(obj):
                names.extend(leaf_names(getattr(obj, field_.name, None),
                                        f"{prefix}{field_.name}."))
            return names
        return [prefix.rstrip(".")]

    leaves = leaf_names(FrameworkConfig())
    field_names = {leaf.split(".")[-1]: leaf for leaf in leaves}

    root = Path(schema_module.__file__).resolve().parents[1]
    readers = {name: [] for name in field_names}
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == Path(schema_module.__file__).resolve():
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Attribute) and node.attr in field_names:
                readers[node.attr].append(f"{path.relative_to(root)}:{node.lineno}")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "getattr" and len(node.args) >= 2
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value in field_names):
                readers[node.args[1].value].append(f"{path.relative_to(root)}:{node.lineno}")

    unread = sorted(field_names[name] for name, sites in readers.items() if not sites)
    assert not unread, (
        "these config fields are read by nothing outside schema.py, so setting them "
        "changes nothing while still validating and serialising:\n  "
        + "\n  ".join(unread)
        + "\nDelete them (and any preset that sets them), or wire them up.")
