"""Evaluation, including the cross-rank reduction of the eval metric."""

from __future__ import annotations

from typing import Any

from ..core.batching import model_inputs
from ..core.tensors import move_to
from ..core.torch import require_torch


def reduce_eval_stats(trainer: Any, local_sum: float, local_count: float) -> tuple[float, float]:
    """Sum (loss, contributor count) across ranks for a global eval metric.

    With ``dp_group=None`` every data-parallel rank iterates the full dataset
    and every pipeline stage but the last yields no loss, so the ranks that
    did measure all hold the same value: averaging over *contributors* is the
    global metric.  Averaging the local mean over ``world_size`` would be
    wrong, and returning the local mean reported NaN on every non-final stage.
    """
    torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
    if trainer.runtime.world_size <= 1:
        return local_sum, local_count
    try:
        import torch.distributed as dist
    except ImportError:
        return local_sum, local_count
    if not (dist.is_available() and dist.is_initialized()):
        return local_sum, local_count
    device = trainer.device if trainer.device.type == "cuda" else torch.device("cpu")
    stats = torch.tensor([local_sum, local_count], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), float(stats[1].item())


def evaluate(trainer: Any, data) -> dict[str, float]:
    torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
    loader = data.train_dataloader(dp_group=None, seed=trainer.config.seed) if hasattr(data, "train_dataloader") else data
    losses: list[float] = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to(batch, trainer.device)
            if bool(getattr(trainer.model.module, "uses_pipeline", False)):
                result = trainer.model.module.pipeline_evaluate(batch, loss_fn=trainer.loss_fn)
                value = result.loss
                if torch.is_tensor(value):
                    losses.append(float(value.float().item()))
                continue
            with trainer._autocast():
                args, kwargs = model_inputs(batch)
                output = trainer.model(*args, **kwargs)
                losses.append(float(trainer._compute_loss(output, batch).detach().float().item()))
    total_sum, total_count = reduce_eval_stats(trainer, float(sum(losses)), float(len(losses)))
    if total_count == 0:
        raise ValueError(
            "evaluate() produced no loss on any rank; pass loss_fn when the model "
            "output carries no loss (a pipeline stage other than the last cannot "
            "compute it itself)")
    return {"loss": total_sum / total_count, "steps": float(len(losses))}
