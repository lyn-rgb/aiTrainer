import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, Trainer


def test_single_process_trainer_checkpoint_continues_step(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, config=FrameworkConfig(device="cpu", seed=7))
    batch = [(torch.randn(4, 3), torch.randn(4, 2))]
    trainer.loss_fn = lambda output, item: torch.nn.functional.mse_loss(output, item[1])
    trainer.fit(batch)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    next_step = trainer.global_step

    model2 = torch.nn.Linear(3, 2)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    trainer2 = Trainer(model2, optimizer2, config=FrameworkConfig(device="cpu", seed=7))
    trainer2.loss_fn = trainer.loss_fn
    trainer2.load_checkpoint(checkpoint)
    assert trainer2.global_step == next_step
    assert trainer2.optimizer_step == trainer.optimizer_step


def test_facade_state_stays_directly_assignable():
    """``Trainer`` must own its state as plain attributes, not proxied properties.

    The step bodies live in sibling modules now (``trainer/step.py`` &co) and
    take the trainer as an argument.  That is only safe because the facade holds
    the state itself: routing it through a session object would turn
    ``trainer.loss_fn = f`` into a silent no-op, and the resulting failure would
    look like a wrong number rather than an AttributeError -- the hardest kind to
    trace back to a refactor.  This test exists so that regression is loud.
    """
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                      config=FrameworkConfig(device="cpu", seed=3))

    def marker(output, item):
        return output.sum() * 0.0

    trainer.loss_fn = marker
    assert trainer.loss_fn is marker, "loss_fn assignment did not stick"

    before = trainer.scaler
    assert "scaler" in vars(trainer), "state moved off the instance"
    trainer.scaler = before
    assert trainer.scaler is before

    # And the delegated step must observe the assigned value, not a stale copy.
    batch = [(torch.randn(4, 3), torch.randn(4, 2))]
    step = trainer.train_step(batch[0])
    assert step.loss == pytest.approx(0.0), "the step used a stale loss_fn"
