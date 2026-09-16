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


def test_manifest_round_trip_keeps_every_field(tmp_path):
    """A field added to Manifest but not to ``from_dict`` disappears SILENTLY.

    This is not hypothetical: ``dp_sharded`` was added to the dataclass and to
    ``to_dict`` (which uses ``asdict``) but not to ``from_dict``, which reads
    fields one by one.  The manifest wrote it, read it back as the default
    ``False``, and the loader then rejected a DP-sharded checkpoint as "converted
    with dp_sharded=False" -- the field was right there in the JSON.

    So this drives the round trip from ``dataclasses.fields`` rather than from a
    hand-written list: a new field fails this test until it is serialized.
    """
    from dataclasses import fields

    from aitrainer.checkpoint.format import Manifest, ShardSpec, load_manifest, write_manifest

    shard = ShardSpec(tensor_name="w", global_shape=(4, 4), dtype="torch.float32",
                      layout="tp_sharded", target_rank=(1, 1, 1), source_file="rank_00001.pt",
                      source_key="w", source_offset=8, source_length=16, local_shape=(2, 4),
                      global_offset=(2, 0), replicated=False, transform="transpose",
                      checksum="deadbeef")
    manifest = Manifest(format_version=1, model_config_hash="mch", tensor_schema_hash="tsh",
                        world_size_at_save=8, logical_sharding={"tp": 2, "pp": 2, "dp": 2},
                        dp_sharded=True, tensors={"w": (shard,)})
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    restored = load_manifest(path)

    for field in fields(Manifest):
        assert getattr(restored, field.name) == getattr(manifest, field.name), (
            f"Manifest.{field.name} did not survive the round trip -- "
            "it is in the dataclass but not in from_dict")
    assert restored.tensors["w"][0] == shard


def test_sharded_checkpoint_restores_the_random_stream(tmp_path):
    """Resuming from a sharded checkpoint must reproduce the run it resumed.

    ``save_sharded`` wrote ``model``/``optimizer``/``bookkeeping`` and nothing
    else, so a resumed run drew different randomness than the run that wrote it
    -- two runs of the same experiment diverged from the resume point, with no
    sign that anything was missing.  The per-rank format had always stored the
    torch and Python RNG state and restored it by default; this is the sharded
    path catching up.
    """
    import random

    import torch

    from aitrainer import FrameworkConfig, Runtime, Trainer

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, value):
            return self.linear(value)

    def build() -> Trainer:
        torch.manual_seed(3)
        runtime = Runtime(device="cpu", init_process_group=False, seed=3)
        model = Model()
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                       config=FrameworkConfig(device="cpu", seed=3), runtime=runtime,
                       loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))

    # Two identical runs.  The first is the reference: it draws what the stream
    # gives after the fit, which is what a resume has to reproduce.  The second
    # saves at the same point and is then moved somewhere else before loading.
    #
    # Drawing `expected` from the SAME run and then saving is the trap -- the
    # saved stream is by then already past those draws, so the comparison fails
    # against a fully working restore.  That version of this test was written
    # first and reported the checkpoint as broken.
    def run() -> Trainer:
        trainer = build()
        torch.manual_seed(555)
        random.seed(555)
        trainer.fit([(torch.randn(2, 8), torch.randn(2, 8))], epochs=1)
        return trainer

    run()
    expected = [torch.randn(3).tolist(), random.random()]

    trainer = run()
    path = tmp_path / "sharded"
    trainer.save_sharded(path)

    # Move both streams somewhere else, then load and see whether they come back.
    torch.manual_seed(991)
    random.seed(991)
    trainer.load_sharded(path)

    got = [torch.randn(3).tolist(), random.random()]
    assert got[0] == expected[0], (
        "the torch stream was not restored; a run resumed from a sharded checkpoint "
        "does not draw what the run that wrote it drew")
    assert got[1] == expected[1], "the Python stream was not restored"
    trainer.close()


def _scheduled_trainer(scheduler_factory, *, provider=None):
    """A trainer whose LR schedule actually moves, for the resume tests below."""
    import torch

    from aitrainer import FrameworkConfig, Runtime, Trainer

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, value):
            return self.linear(value)

    torch.manual_seed(3)
    runtime = Runtime(device="cpu", init_process_group=False, seed=3)
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, config=FrameworkConfig(device="cpu", seed=3),
                      runtime=runtime, scheduler=scheduler_factory(optimizer),
                      loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    if provider is not None:
        trainer._data_provider = provider
    return trainer


def _batches(count):
    import torch
    return [(torch.randn(2, 8), torch.randn(2, 8)) for _ in range(count)]


def test_sharded_checkpoint_restores_the_scheduler(tmp_path):
    """A resume must land on the same LR, not restart the schedule.

    ``save_sharded`` wrote model/optimizer/bookkeeping, so
    ``scheduler.state_dict()`` was simply absent: the optimizer's own
    ``param_groups`` carried the current lr back -- which is why this went
    unnoticed -- but ``last_epoch`` came back 0 and the schedule restarted from
    there.  Measured with ``CosineAnnealingLR``, which computes an lr from
    ``last_epoch``: after four steps the run sits at 0.065451, and two more
    steps gave 0.059201 from a sharded resume against the correct 0.034549.

    ``StepLR`` is the reason it hid: it *multiplies* whatever lr it finds, so
    restoring the optimizer alone happens to reproduce the schedule exactly.
    """
    import torch

    for name, factory in (
            ("cosine", lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)),
            ("step", lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5))):
        path = tmp_path / name

        reference = _scheduled_trainer(factory)
        for batch in _batches(4):
            reference.train_step(batch)
        reference.save_checkpoint(tmp_path / f"{name}-per-rank")

        saver = _scheduled_trainer(factory)
        for batch in _batches(4):
            saver.train_step(batch)
        saver.save_sharded(path)

        # Per-rank format: the reference the sharded one has to match.
        from_per_rank = _scheduled_trainer(factory)
        from_per_rank.load_checkpoint(tmp_path / f"{name}-per-rank")
        from_sharded = _scheduled_trainer(factory)
        from_sharded.load_sharded(path)

        assert (from_sharded.scheduler.last_epoch
                == from_per_rank.scheduler.last_epoch
                == reference.scheduler.last_epoch), (
            f"{name}: the schedule position did not survive a sharded resume "
            f"(sharded {from_sharded.scheduler.last_epoch}, "
            f"per-rank {from_per_rank.scheduler.last_epoch})")
        # And the lr those positions produce, which is what training actually
        # consumes: two more steps must follow the same curve.
        for trainer in (from_per_rank, from_sharded):
            for batch in _batches(2):
                trainer.train_step(batch)
        assert (from_sharded.optimizer.param_groups[0]["lr"]
                == from_per_rank.optimizer.param_groups[0]["lr"]), (
            f"{name}: the lr schedule diverged after a sharded resume "
            f"({from_sharded.optimizer.param_groups[0]['lr']} vs "
            f"{from_per_rank.optimizer.param_groups[0]['lr']})")
        for trainer in (reference, saver, from_per_rank, from_sharded):
            trainer.close()


def test_sharded_checkpoint_restores_the_data_position(tmp_path):
    """The provider's read position travels too, and is applied by the next fit.

    ``load_sharded`` cannot apply it: the provider is ``fit``'s ``data``
    argument, which is passed after the checkpoint is loaded.  It is stashed and
    taken by exactly one ``fit`` -- leaving it set would rewind the provider on
    every later call.
    """
    import torch

    class Provider:
        """Only the two methods the loop contract uses."""

        def __init__(self, cursor: int = 0) -> None:
            self.cursor = cursor
            self.applied = []

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.applied.append(dict(state))
            self.cursor = state["cursor"]

        def train_dataloader(self, *, dp_group=None, seed=None):
            return [(torch.randn(2, 8), torch.randn(2, 8))]

    saver = _scheduled_trainer(lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1),
                               provider=Provider(cursor=17))
    path = tmp_path / "sharded"
    saver.save_sharded(path)

    resumed = _scheduled_trainer(lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1))
    provider = Provider(cursor=0)
    resumed.load_sharded(path)
    assert provider.applied == [], "the provider was loaded before fit had it"

    resumed.fit(provider, epochs=1)
    assert provider.applied == [{"cursor": 17}], (
        f"the saved position did not reach the provider: {provider.applied}")

    # A second fit must not rewind it.
    resumed.fit(provider, epochs=1)
    assert provider.applied == [{"cursor": 17}], (
        "the stashed position was applied more than once, so every later fit "
        f"rewinds the data: {provider.applied}")
    for trainer in (saver, resumed):
        trainer.close()


def test_sharded_load_refuses_a_namespace_it_cannot_restore(tmp_path):
    """A checkpoint holding more than the recipe claims must be refused, not half-read.

    DCP fills the state dict it is handed and ignores the rest, so a scheduler
    saved but not claimed restores everything else and drops it silently --
    which is what the sharded format did to the RNG and the schedule.  The
    reverse direction is already loud (DCP raises "Missing key").

    The check is per namespace rather than per key on purpose: a sharded
    checkpoint is partitioned across ranks while the metadata is global, so the
    unclaimed *keys* on one rank are mostly another rank's (a pipeline stage's
    other stages, the RNG under another coordinate).  Comparing keys refused
    every working checkpoint -- measured, 48 spurious entries at dp2-pp2.
    """
    import pytest
    import torch

    from aitrainer.checkpoint.manager import CheckpointError

    saver = _scheduled_trainer(lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1))
    path = tmp_path / "sharded"
    saver.save_sharded(path)

    # The same checkpoint, restored by a trainer with no scheduler: the
    # ``scheduler`` namespace is in the file and nothing would read it.
    unscheduled = _scheduled_trainer(lambda opt: torch.optim.lr_scheduler.StepLR(opt, 1))
    unscheduled.scheduler = None
    with pytest.raises(CheckpointError) as caught:
        unscheduled.load_sharded(path)
    assert "scheduler" in str(caught.value), (
        f"the refusal does not name the namespace it would drop: {caught.value}")

    for trainer in (saver, unscheduled):
        trainer.close()


def test_flatten_keys_matches_torch(tmp_path):
    """``_flatten_keys`` must mirror DCP's own key space, not approximate it.

    The manager reimplements the traversal in ``torch/.../_traverse.py`` rather
    than importing that private module, and the two are pinned together here
    against the installed torch.  A version that changed the rule fails this
    test instead of quietly weakening the namespace check above.

    The interesting case is a list: DCP descends into one only when it holds
    something traversable, so ``base_lrs=[0.1, 0.1]`` is ONE key while
    ``[{"a": 1}]`` is ``....0.a``.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader, save

    from aitrainer.checkpoint.manager import CheckpointManager

    cases = {
        "plain": {"w": torch.zeros(2)},
        "nested": {"block": {"w": torch.zeros(2), "b": torch.zeros(2)}},
        "list of numbers": {"nested": {"base_lrs": [0.1, 0.1], "_last_lr": [0.03]}},
        "list of mappings": {"nested": {"groups": [{"lr": 0.1}, {"lr": 0.2}]}},
        "list of tensors": {"nested": {"bufs": [torch.zeros(2), torch.zeros(3)]}},
        "mixed": {"a": {"b": [{"c": torch.zeros(1)}]}, "d": 3},
    }
    root = tmp_path / "probe"
    for index, (label, value) in enumerate(cases.items()):
        target = root / str(index)
        save(dict(value), checkpoint_id=str(target))
        theirs = set(FileSystemReader(str(target)).read_metadata().state_dict_metadata)
        ours = CheckpointManager._flatten_keys(value)
        assert ours == theirs, (
            f"{label}: _flatten_keys disagrees with torch's traversal "
            f"(ours {sorted(ours)}, torch {sorted(theirs)})")


def test_sharded_rng_carries_the_cuda_generators(monkeypatch):
    """A GPU resume has to restore the CUDA generator, not just the CPU one.

    The per-rank format always stored both (``_torch_rng_state`` in
    ``checkpoint/manager.py``); the sharded path stored only the CPU generator
    and Python's, so on a GPU box a resumed run drew different dropout masks than
    the run that wrote the checkpoint -- and a CPU-only host cannot see it,
    because ``is_available()`` is False and there is simply no CUDA generator to
    miss.  That is why this is a structural test: with no accelerator here, the
    observable facts are which calls happen.  The numbers can only be checked on
    a GPU box -- see the GPU verification checklist in docs/代码审计报告.md.
    """
    import torch

    from aitrainer.trainer.checkpointing import _restore_rng, _rng_state

    wanted = [torch.arange(8, dtype=torch.uint8), torch.arange(8, dtype=torch.uint8) + 1]
    applied: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [row.clone() for row in wanted])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all",
                        lambda states: applied.append([row.tolist() for row in states]))

    state = _rng_state()
    assert "cuda" in state, (
        "the CUDA generators were not saved, so a GPU resume diverges from the run "
        "it resumed and nothing says so")
    assert tuple(state["cuda"].shape) == (2, 8), "one row per device, stacked"

    _restore_rng(state)
    assert applied == [[row.tolist() for row in wanted]], (
        "the saved CUDA state was never handed back to torch")
