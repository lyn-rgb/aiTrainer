"""One training step, in all three shapes the trainer needs.

These are free functions over a ``Trainer`` rather than methods on a session
object, and that is deliberate: ``test_audit_regressions`` and
``test_trainer_contract`` assign ``trainer.loss_fn`` and ``trainer.scaler``
directly.  A facade that proxied its state would turn those assignments into
silent no-ops, and the resulting failures would look numerical rather than
structural -- the worst kind to debug.  The facade therefore owns its state as
plain attributes and these functions read it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..core.batching import model_inputs
from ..core.tensors import move_to
from ..core.torch import require_torch
from ..precision import cast_gradients
from .state import StepOutput


def run_optimizer_step(trainer: Any) -> tuple[float | None, bool]:
    """Unscale, cast, clip, step, and advance bookkeeping for one update.

    Shared by both training paths so they cannot drift apart.  Previously each
    site carried its own copy: one omitted gradient clipping entirely, and none
    respected a GradScaler skip -- so an inf/nan spike still advanced the LR
    schedule and incremented ``optimizer_step`` for an update that never
    happened.

    Returns ``(grad_norm, applied)``; ``applied`` is False when the scaler
    skipped the update, so callers can report the step honestly instead of
    claiming an update that did not occur.
    """
    torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
    if trainer.scaler.is_enabled():
        trainer.scaler.unscale_(trainer.optimizer)
    # Sequence parallelism shards each norm's activation, so the norm's
    # replicated parameters accumulate a Partial (per-rank partial sum)
    # gradient that nothing reduces.  Doing it here -- before clipping and
    # before the optimizer -- is what makes the momentum buffers correct and
    # rank-consistent, and what stops gradient clipping from taking the norm of
    # a partial sum.  See parallel.tp.reduce_replicated_gradients; it is a
    # no-op for every model with no Partial gradients.
    from ..parallel.tp import reduce_replicated_gradients
    reduce_replicated_gradients(trainer.model.module)
    cast_gradients(trainer.model.parameters(), trainer.config.precision.grad_dtype, torch_module=torch)
    grad_norm = None
    if trainer.config.grad_clip_norm is not None:
        # FSDP1 exposed `module.clip_grad_norm_`; FSDP2 does not, and
        # `torch.distributed.fsdp.clip_grad_norm_` does not exist in torch 2.9.
        # `parallel.fsdp.clip_grad_norm_` reduces the per-parameter DTensor norms
        # itself: `torch.nn.utils` does not, and returns each shard's own norm.
        from ..parallel.fsdp import clip_grad_norm_
        value = clip_grad_norm_(trainer.model.module, trainer.config.grad_clip_norm)
        grad_norm = float(value.item())
    skipped = False
    if trainer.scaler.is_enabled():
        scale_before = trainer.scaler.get_scale()
        trainer.offload.step_optimizer(trainer.optimizer, lambda: trainer.scaler.step(trainer.optimizer),
                                       optimizer_dtype=trainer.config.precision.optimizer_dtype)
        trainer.scaler.update()
        # update() lowers the scale exactly when inf/nan was detected and the
        # step was skipped; GradScaler exposes no public skip flag.  A scale
        # pinned at 0.0 -- torch stops lowering it after ~167 consecutive
        # found_inf updates -- can no longer decrease, so count that as a skip.
        after = trainer.scaler.get_scale()
        skipped = after < scale_before or after == 0.0
    else:
        trainer.offload.step_optimizer(trainer.optimizer, trainer.optimizer.step,
                                       optimizer_dtype=trainer.config.precision.optimizer_dtype)
    trainer.optimizer.zero_grad(set_to_none=True)
    if not skipped:
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        trainer.optimizer_step += 1
    return grad_norm, not skipped


def run_pipeline_step(trainer: Any, batch: Any) -> StepOutput:
    """Run a PP-owned backward step and update the stage optimizer once."""
    torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
    batch = move_to(batch, trainer.device)
    will_step = (trainer.global_step + 1) % trainer.config.grad_accumulation_steps == 0
    sync_context = nullcontext() if will_step else trainer.model.no_sync()
    trainer.offload.fetch(trainer.model.module, device=trainer.device)
    try:
        with trainer.offload.activation_context(trainer.device), sync_context, trainer._autocast():
            result = trainer.model.module.pipeline_step(
                batch, loss_fn=trainer.loss_fn, scaler=trainer.scaler,
                accumulation_steps=trainer.config.grad_accumulation_steps)
        loss_value = result.loss          # ScheduleOutput, not a dict
        raw_loss = float(loss_value.float().item()) if torch.is_tensor(loss_value) else 0.0
        trainer.global_step += 1
        should_step = trainer.global_step % trainer.config.grad_accumulation_steps == 0
        grad_norm = None
        applied = False          # no update happens until the window completes
        if should_step:
            grad_norm, applied = run_optimizer_step(trainer)
        output = StepOutput(raw_loss, trainer.global_step, applied, grad_norm)
        trainer.history.append(output)
        return output
    finally:
        trainer.offload.release(trainer.model.module)


def run_train_step(trainer: Any, batch: Any) -> StepOutput:
    torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
    trainer.model.train()
    if bool(getattr(trainer.model.module, "uses_pipeline", False)):
        return run_pipeline_step(trainer, batch)
    batch = move_to(batch, trainer.device)
    # no_sync() must span forward AND backward: FSDP's all-gather and DDP's
    # reduction hooks are installed during forward.  Skipping it made every
    # microbatch pay a full gradient reduction instead of one per
    # accumulation window -- the only reason this guard exists.
    will_step = (trainer.global_step + 1) % trainer.config.grad_accumulation_steps == 0
    sync_context = nullcontext() if will_step else trainer.model.no_sync()
    trainer.offload.fetch(trainer.model.module, device=trainer.device)
    try:
        with trainer.offload.activation_context(trainer.device), sync_context, trainer._autocast():
            args, kwargs = model_inputs(batch)
            output = trainer.model(*args, **kwargs)
            loss = trainer._compute_loss(output, batch)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar torch.Tensor")
            raw_loss = loss.detach().float().item()
            scaled_loss = loss / trainer.config.grad_accumulation_steps
            if trainer.scaler.is_enabled():
                trainer.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
    except BaseException:
        # Discard the partial window.  Its earlier microbatches are still in
        # .grad, so a caller that catches the failure and retries the window
        # would accumulate them a second time; partial windows are dropped
        # rather than flushed, so nothing of value is lost.
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.offload.release(trainer.model.module)
        raise
    try:
        trainer.global_step += 1
        should_step = trainer.global_step % trainer.config.grad_accumulation_steps == 0
        grad_norm = None
        applied = False          # no update happens until the window completes
        if should_step:
            grad_norm, applied = run_optimizer_step(trainer)
        result = StepOutput(raw_loss, trainer.global_step, applied, grad_norm)
        trainer.history.append(result)
        return result
    finally:
        # Release only after the optimizer has consumed device parameters.
        trainer.offload.release(trainer.model.module)
