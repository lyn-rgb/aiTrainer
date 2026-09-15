"""Forward/backward correctness gates for the eager and offload paths."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from aitrainer import FrameworkConfig, OffloadConfig, Trainer  # noqa: E402


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Tanh(), torch.nn.Linear(24, 5))


def _loss(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(model(inputs), labels)


def test_forward_backward_matches_identical_eager_clones():
    torch.manual_seed(2026)
    baseline = _model()
    candidate = _model()
    candidate.load_state_dict(baseline.state_dict())
    inputs = torch.randn(16, 12)
    labels = torch.randint(0, 5, (16,))
    baseline_loss = _loss(baseline, inputs, labels)
    candidate_loss = _loss(candidate, inputs, labels)
    baseline_loss.backward()
    candidate_loss.backward()
    torch.testing.assert_close(candidate_loss, baseline_loss, rtol=0.0, atol=0.0)
    for expected, actual in zip(baseline.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0.0, atol=0.0)
        assert torch.isfinite(actual.grad).all()


def test_trainer_step_updates_parameters_and_returns_finite_loss():
    torch.manual_seed(2027)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [item.detach().clone() for item in model.parameters()]
    trainer = Trainer(model, optimizer, config=FrameworkConfig(device="cpu"),
                      loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1]))
    inputs = torch.randn(16, 12)
    labels = torch.randint(0, 5, (16,))
    result = trainer.train_step((inputs, labels))
    assert math.isfinite(result.loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    trainer.close()


def test_optimizer_offload_preserves_one_step_result():
    torch.manual_seed(2028)
    baseline = _model()
    offloaded = _model()
    offloaded.load_state_dict(baseline.state_dict())
    input_value = torch.randn(16, 12)
    labels = torch.randint(0, 5, (16,))
    opt_a = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    opt_b = torch.optim.AdamW(offloaded.parameters(), lr=1e-3)
    cfg = FrameworkConfig(device="cpu", offload=OffloadConfig(enabled=True, optimizer=True))
    trainer = Trainer(offloaded, opt_b, config=cfg,
                      loss_fn=lambda output, batch: torch.nn.functional.cross_entropy(output, batch[1]))
    loss_a = _loss(baseline, input_value, labels)
    loss_a.backward()
    opt_a.step()
    opt_a.zero_grad(set_to_none=True)
    trainer.train_step((input_value, labels))
    for expected, actual in zip(baseline.parameters(), offloaded.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    trainer.close()

