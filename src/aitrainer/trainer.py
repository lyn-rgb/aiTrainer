"""Correctness-first single-process training loop for Batch 0."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import os
import warnings
from typing import Any, Callable, Iterable, Mapping

from .adapters import DataProvider, LossFn
from .capability import validate_capabilities
from .config import FrameworkConfig
from .data import TARGET_KEYS, model_inputs
from .checkpoint.manager import CheckpointManager
from .distributed_model import DistributedModel
from .runtime import Runtime
from .offload import OffloadManager
from .overlap import OverlapController
from .precision import cast_gradients, validate_precision

logger = logging.getLogger("aitrainer")


@dataclass(frozen=True)
class StepOutput:
    loss: float
    step: int
    optimizer_step: bool
    grad_norm: float | None = None


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("aiTrainer training requires PyTorch; install the project's torch dependency") from exc
    return torch


def _move(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


class Trainer:
    """Train a regular PyTorch module with AMP, accumulation and checkpoints."""

    def __init__(self, model: Any, optimizer: Any, *, config: FrameworkConfig | None = None,
                 loss_fn: LossFn | Callable[[Any, Any], Any] | None = None,
                 scheduler: Any | None = None, device: str | None = None,
                 runtime: Runtime | None = None) -> None:
        """Wrap an already-built model.

        The seed is applied when the runtime is created, which is here -- so a
        model the caller built beforehand has already drawn its initialisation
        from the ambient RNG.  For reproducible weight init use
        :meth:`from_model`, or create a :class:`Runtime` (which seeds) before
        building the model.
        """
        torch = _import_torch()
        self.config = config or FrameworkConfig()
        self.runtime = runtime or Runtime(device=device or self.config.device, seed=self.config.seed)
        self.config.validate(world_size=self.runtime.world_size)
        # Capability gating must not depend on the entry point.  Trainer.from_model
        # routes through parallelize() (which validates), but constructing a Trainer
        # directly skipped it -- so FSDP + gradient-bucket overlap was accepted
        # silently here while dry_run/from_model rejected the same config.
        validate_capabilities(self.config, world_size=self.runtime.world_size, torch_module=torch)
        if self.runtime.world_size > 1 and self.config.parallel.dp_size > 1:
            # Only REPLICATED ranks need to reduce gradients together.  TP/PP with
            # dp_size == 1 has no replicas to diverge, and TP already reduces in the
            # autograd collectives -- demanding no_sync there made the STABLE `tp`
            # capability impossible to train through Trainer.  Inspect the layer
            # that actually reduces, not the wrapper: DistributedModel.no_sync()
            # degrades to nullcontext() when the module it wraps has none, so
            # testing the wrapper let an unsynchronised model pass this guard.
            wrapped = model.module if isinstance(model, DistributedModel) else model
            if not hasattr(wrapped, "no_sync"):
                raise ValueError(
                    "dp_size>1 requires a model that reduces gradients (an FSDP-wrapped "
                    "module): the given model has no no_sync, so each rank would step on "
                    "its own divergent gradients. Set fsdp.enabled=True, or use dp_size=1 "
                    "with tp_size/pp_size for the non-replicated axes.")
        self.model = model if isinstance(model, DistributedModel) else DistributedModel(model)
        self.device = torch.device(device or self.runtime.state.device)
        validate_precision(self.config.precision, device=self.device, torch_module=torch)
        self.model.module.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        use_scaler = self.config.precision.use_grad_scaler
        compute_name = self._dtype_name(self.config.precision.compute_dtype)
        # A reduced-precision compute dtype only takes effect on CUDA (see
        # _autocast).  On CPU it silently runs fp32; previously the only check was
        # an unreachable branch in capability.py, so nothing told the user.
        if compute_name in {"float16", "bfloat16"} and self.device.type != "cuda":
            warnings.warn(
                f"precision.compute_dtype={compute_name!r} has no effect on device "
                f"{self.device.type!r}; training runs in fp32",
                RuntimeWarning, stacklevel=2)
        if use_scaler is None:
            use_scaler = compute_name == "float16"
        enabled = bool(use_scaler and self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        self.global_step = 0
        self.optimizer_step = 0
        self.history: list[StepOutput] = []
        self._data_provider: Any = None
        self.offload = OffloadManager(self.config.offload, overlap_config=self.config.overlap)
        self.overlap = OverlapController(enabled=any((self.config.overlap.enable_tp_bulk_overlap,
                                                     self.config.overlap.enable_gradient_bucket_overlap,
                                                     self.config.overlap.enable_parameter_prefetch,
                                                     self.config.overlap.enable_pp_p2p_overlap,
                                                     self.config.overlap.enable_transfer_overlap)),
                                         max_inflight_ops=self.config.overlap.max_inflight_ops,
                                         max_inflight_bytes=self.config.overlap.max_inflight_bytes)
        self.offload.register_model(self.model.module)
        self.offload.register_optimizer(self.optimizer)

    @classmethod
    def from_model(cls, model_or_adapter: Any, *, config: FrameworkConfig | None = None,
                   optimizer: Any | None = None, optimizer_factory: Any | None = None,
                   optimizer_cls: Any | None = None, optimizer_kwargs: Mapping[str, Any] | None = None,
                   loss_fn: LossFn | Callable[[Any, Any], Any] | None = None,
                   scheduler: Any | None = None, device: str | None = None,
                   runtime: Runtime | None = None) -> "Trainer":
        """Build a trainer from a model or adapter without hiding parallel choices."""
        model = model_or_adapter
        runtime_obj = runtime
        requested_parallel = bool(config is not None and (
            config.fsdp.enabled or config.parallel.dp_size > 1 or
            config.parallel.tp_size > 1 or config.parallel.pp_size > 1))
        # Create the runtime -- and therefore apply the seed -- BEFORE the model
        # is built.  Building first drew weight init from the ambient unseeded
        # RNG, so `seed` never made initialisation reproducible.
        if runtime_obj is None and config is not None:
            runtime_obj = Runtime(device=device or config.device, seed=config.seed)
        if hasattr(model_or_adapter, "build") and not hasattr(model_or_adapter, "parameters"):
            build_device = device or (config.device if config is not None else "auto")
            if build_device == "auto":
                build_device = "cpu"
            model = model_or_adapter.build(device=build_device)
        if requested_parallel:
            if optimizer is not None:
                raise ValueError("pass an optimizer factory/class when automatic parallel wrapping is enabled")
            from .parallelizer import parallelize
            model = parallelize(model, config=config, runtime=runtime_obj)
        if optimizer is not None and (optimizer_factory is not None or optimizer_cls is not None):
            raise ValueError("pass only one of optimizer, optimizer_factory, or optimizer_cls")
        if optimizer is None and optimizer_factory is not None:
            optimizer = optimizer_factory(model.parameters()) if callable(optimizer_factory) else optimizer_factory.build(model.parameters())
        if optimizer is None:
            torch = _import_torch()
            cls_optimizer = optimizer_cls or torch.optim.AdamW
            kwargs = dict(optimizer_kwargs or {})
            kwargs.setdefault("lr", 1e-3)
            optimizer = cls_optimizer(model.parameters(), **kwargs)
        return cls(model, optimizer, config=config, loss_fn=loss_fn, scheduler=scheduler,
                   device=device, runtime=runtime_obj)

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        return str(dtype).replace("torch.", "").lower()

    def _autocast(self):
        torch = _import_torch()
        name = self._dtype_name(self.config.precision.compute_dtype)
        if self.device.type == "cuda" and name in {"float16", "bfloat16"}:
            dtype = torch.float16 if name == "float16" else torch.bfloat16
            try:
                return torch.autocast(device_type="cuda", dtype=dtype)
            except TypeError:
                return torch.cuda.amp.autocast(dtype=dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def _compute_loss(self, output: Any, batch: Any) -> Any:
        if self.loss_fn is not None:
            return self.loss_fn(output, batch)
        if isinstance(output, Mapping) and "loss" in output:
            return output["loss"]
        if hasattr(output, "loss"):
            return output.loss
        logits = output["logits"] if isinstance(output, Mapping) and "logits" in output else output
        labels = None
        if isinstance(batch, Mapping):
            # Read the same target keys model_inputs strips, so the strip-set and
            # the read-set cannot disagree (a batch using "target" used to raise
            # even though the error message promised a Mapping carrying labels).
            for key in TARGET_KEYS:
                if key in batch:
                    labels = batch[key]
                    break
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            labels = batch[1]
        if labels is None:
            raise TypeError(
                "loss_fn is required unless the model output contains 'loss' or the batch carries "
                "labels (a Mapping with 'labels', or a tuple whose second element is the target)"
            )
        # Flatten sequence dims so [N, T, C] logits pair with [N, T] labels.
        return _import_torch().nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    def _optimizer_step_block(self) -> float | None:
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
        torch = _import_torch()
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        cast_gradients(self.model.parameters(), self.config.precision.grad_dtype, torch_module=torch)
        grad_norm = None
        if self.config.grad_clip_norm is not None:
            if hasattr(self.model.module, "clip_grad_norm_"):
                value = self.model.module.clip_grad_norm_(self.config.grad_clip_norm)
            else:
                value = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            grad_norm = float(value.item())
        skipped = False
        if self.scaler.is_enabled():
            scale_before = self.scaler.get_scale()
            self.offload.step_optimizer(self.optimizer, lambda: self.scaler.step(self.optimizer),
                                        optimizer_dtype=self.config.precision.optimizer_dtype)
            self.scaler.update()
            # update() lowers the scale exactly when inf/nan was detected and the
            # step was skipped; GradScaler exposes no public skip flag.  A scale
            # pinned at 0.0 -- torch stops lowering it after ~167 consecutive
            # found_inf updates -- can no longer decrease, so count that as a skip.
            after = self.scaler.get_scale()
            skipped = after < scale_before or after == 0.0
        else:
            self.offload.step_optimizer(self.optimizer, self.optimizer.step,
                                        optimizer_dtype=self.config.precision.optimizer_dtype)
        self.optimizer.zero_grad(set_to_none=True)
        if not skipped:
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer_step += 1
        return grad_norm, not skipped

    def _pipeline_train_step(self, batch: Any) -> StepOutput:
        """Run a PP-owned backward step and update the stage optimizer once."""
        torch = _import_torch()
        batch = _move(batch, self.device)
        will_step = (self.global_step + 1) % self.config.grad_accumulation_steps == 0
        sync_context = nullcontext() if will_step else self.model.no_sync()
        self.offload.fetch(self.model.module, device=self.device)
        try:
            with self.offload.activation_context(self.device), sync_context, self._autocast():
                result = self.model.module.pipeline_step(
                    batch, loss_fn=self.loss_fn, scaler=self.scaler,
                    accumulation_steps=self.config.grad_accumulation_steps)
            loss_value = result.get("loss")
            raw_loss = float(loss_value.float().item()) if torch.is_tensor(loss_value) else 0.0
            self.global_step += 1
            should_step = self.global_step % self.config.grad_accumulation_steps == 0
            grad_norm = None
            applied = False          # no update happens until the window completes
            if should_step:
                grad_norm, applied = self._optimizer_step_block()
            output = StepOutput(raw_loss, self.global_step, applied, grad_norm)
            self.history.append(output)
            return output
        finally:
            self.offload.release(self.model.module)

    def train_step(self, batch: Any) -> StepOutput:
        torch = _import_torch()
        self.model.train()
        if bool(getattr(self.model.module, "uses_pipeline", False)):
            return self._pipeline_train_step(batch)
        batch = _move(batch, self.device)
        # no_sync() must span forward AND backward: FSDP's all-gather and DDP's
        # reduction hooks are installed during forward.  Skipping it made every
        # microbatch pay a full gradient reduction instead of one per
        # accumulation window -- the only reason this guard exists.
        will_step = (self.global_step + 1) % self.config.grad_accumulation_steps == 0
        sync_context = nullcontext() if will_step else self.model.no_sync()
        self.offload.fetch(self.model.module, device=self.device)
        try:
            with self.offload.activation_context(self.device), sync_context, self._autocast():
                args, kwargs = model_inputs(batch)
                output = self.model(*args, **kwargs)
                loss = self._compute_loss(output, batch)
                if not torch.is_tensor(loss) or loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar torch.Tensor")
                raw_loss = loss.detach().float().item()
                scaled_loss = loss / self.config.grad_accumulation_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        except BaseException:
            # Discard the partial window.  Its earlier microbatches are still in
            # .grad, so a caller that catches the failure and retries the window
            # would accumulate them a second time; partial windows are dropped
            # rather than flushed, so nothing of value is lost.
            self.optimizer.zero_grad(set_to_none=True)
            self.offload.release(self.model.module)
            raise
        try:
            self.global_step += 1
            should_step = self.global_step % self.config.grad_accumulation_steps == 0
            grad_norm = None
            applied = False          # no update happens until the window completes
            if should_step:
                grad_norm, applied = self._optimizer_step_block()
            result = StepOutput(raw_loss, self.global_step, applied, grad_norm)
            self.history.append(result)
            return result
        finally:
            # Release only after the optimizer has consumed device parameters.
            self.offload.release(self.model.module)

    def fit(self, data: Iterable[Any] | DataProvider, *, epochs: int = 1,
            max_steps: int | None = None, resume_from: str | os.PathLike[str] | None = None) -> list[StepOutput]:
        self._data_provider = data if hasattr(data, "state_dict") else None
        if resume_from is not None:
            state = self.load_checkpoint(resume_from)
            if self._data_provider is not None and state.get("sampler_state"):
                self._data_provider.load_state_dict(state["sampler_state"])
        for epoch in range(epochs):
            if hasattr(data, "set_epoch"):
                data.set_epoch(epoch)
            loader = (data.train_dataloader(dp_group=None, seed=self.config.seed)
                      if hasattr(data, "train_dataloader") else data)
            logger.info("starting epoch with global_step=%d", self.global_step)
            for batch in loader:
                self.train_step(batch)
                if self.global_step % self.config.log_interval == 0:
                    logger.info("step=%d loss=%.6f", self.global_step, self.history[-1].loss)
                if max_steps is not None and self.global_step >= max_steps:
                    # Drop the partial accumulation: leaving it in .grad let a
                    # later fit()/train_step() fold it into its first update.
                    self.optimizer.zero_grad(set_to_none=True)
                    return self.history
        # A trailing partial accumulation window is DROPPED, not flushed.  A flush
        # cannot be made correct: the tail microbatches were accumulated under
        # no_sync(), so with dp_size>1 their gradients were never reduced -- a
        # synced backward is what materialises the reduction, and there is no
        # forward left to re-run -- and they are scaled k/G where the window mean
        # is k/k.  Stepping on them both diverged the DP replicas and mis-scaled
        # the update; with nothing accumulated at all it advanced the LR schedule
        # for an update that never happened.  Zeroing keeps the tail out of any
        # later fit()/train_step(), matching the max_steps path above.
        self.optimizer.zero_grad(set_to_none=True)
        return self.history

    def evaluate(self, data: Iterable[Any] | DataProvider) -> dict[str, float]:
        torch = _import_torch()
        loader = data.train_dataloader(dp_group=None, seed=self.config.seed) if hasattr(data, "train_dataloader") else data
        losses: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move(batch, self.device)
                if bool(getattr(self.model.module, "uses_pipeline", False)):
                    result = self.model.module.pipeline_evaluate(batch, loss_fn=self.loss_fn)
                    value = result.get("loss")
                    if torch.is_tensor(value):
                        losses.append(float(value.float().item()))
                    continue
                with self._autocast():
                    args, kwargs = model_inputs(batch)
                    output = self.model(*args, **kwargs)
                    losses.append(float(self._compute_loss(output, batch).detach().float().item()))
        total_sum, total_count = self._reduce_eval_stats(float(sum(losses)), float(len(losses)))
        if total_count == 0:
            raise ValueError(
                "evaluate() produced no loss on any rank; pass loss_fn when the model "
                "output carries no loss (a pipeline stage other than the last cannot "
                "compute it itself)")
        return {"loss": total_sum / total_count, "steps": float(len(losses))}

    def _reduce_eval_stats(self, local_sum: float, local_count: float) -> tuple[float, float]:
        """Sum (loss, contributor count) across ranks for a global eval metric.

        With ``dp_group=None`` every data-parallel rank iterates the full dataset
        and every pipeline stage but the last yields no loss, so the ranks that
        did measure all hold the same value: averaging over *contributors* is the
        global metric.  Averaging the local mean over ``world_size`` would be
        wrong, and returning the local mean reported NaN on every non-final stage.
        """
        torch = _import_torch()
        if self.runtime.world_size <= 1:
            return local_sum, local_count
        try:
            import torch.distributed as dist
        except ImportError:
            return local_sum, local_count
        if not (dist.is_available() and dist.is_initialized()):
            return local_sum, local_count
        device = self.device if self.device.type == "cuda" else torch.device("cpu")
        stats = torch.tensor([local_sum, local_count], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return float(stats[0].item()), float(stats[1].item())

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        self.overlap.drain(self.config.overlap.drain_timeout_s)
        self.offload.drain(self.optimizer)
        rank_mapping = {"rank": self.runtime.rank, "world_size": self.runtime.world_size}
        module = self.model.module
        for attribute, key in (("stage_id", "pp_rank"), ("tp_rank", "tp_rank"), ("dp_rank", "dp_rank"),
                               ("pp_size", "pp_size")):
            if hasattr(module, attribute):
                rank_mapping[key] = int(getattr(module, attribute))
        CheckpointManager(runtime=self.runtime).save(
            path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, global_step=self.global_step, optimizer_step=self.optimizer_step,
            config=self.config.to_dict(), rank_mapping=rank_mapping,
            sampler_state=self._data_provider.state_dict() if self._data_provider is not None else None)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        self.overlap.drain(self.config.overlap.drain_timeout_s)
        self.offload.drain(self.optimizer)
        state = CheckpointManager(runtime=self.runtime).load(
            path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, expected_world_size=self.runtime.world_size)
        self.global_step = int(state["global_step"])
        self.optimizer_step = int(state["optimizer_step"])
        return state

    def close(self) -> None:
        self.overlap.drain(self.config.overlap.drain_timeout_s)
        self.offload.close(self.optimizer)
        self.runtime.close()
