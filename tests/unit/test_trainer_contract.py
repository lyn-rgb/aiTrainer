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
