"""Gradient-accumulation scaling on the pipeline path.

Regression: ``Trainer._pipeline_train_step`` never divided the loss by
``grad_accumulation_steps``, so N accumulated steps were SUMMED and applied at
once — an effective learning rate N times the identical non-pipeline run, with
no error.  The non-pipeline path divides (``trainer.py``), and the documented
contract is ``loss = output.loss / grad_accumulation_steps``.

``PipelineStage`` with ``pp_size == 1`` runs the single-process branch of
``pipeline_step``, so this is testable without any multi-rank rendezvous.
"""

import pytest

torch = pytest.importorskip("torch")

from aitrainer.distributed_model import PipelineStage  # noqa: E402


def _module() -> torch.nn.Module:
    torch.manual_seed(11)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))


def _grads(accumulation_steps: int, inputs: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
    module = _module()
    stage = PipelineStage(module, stage_id=0, pp_size=1)
    stage.pipeline_step((inputs, targets),
                        loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]),
                        accumulation_steps=accumulation_steps)
    return [parameter.grad.detach().clone() for parameter in module.parameters()]


def test_pipeline_step_divides_gradient_by_accumulation_steps():
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    single = _grads(1, inputs, targets)
    for steps in (2, 4):
        accumulated = _grads(steps, inputs, targets)
        for one, many in zip(single, accumulated):
            torch.testing.assert_close(many, one / steps, rtol=1e-6, atol=1e-8)
            assert torch.isfinite(many).all()


def test_pipeline_step_reports_the_unscaled_loss():
    """The logged loss must stay the true mean, not the accumulation-scaled value."""
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    def reported(accumulation_steps: int) -> float:
        module = _module()
        stage = PipelineStage(module, stage_id=0, pp_size=1)
        result = stage.pipeline_step(
            (inputs, targets),
            loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]),
            accumulation_steps=accumulation_steps)
        return float(result.loss)

    assert reported(1) == pytest.approx(reported(4), rel=1e-6)
