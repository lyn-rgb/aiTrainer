"""Uniform wrapper for the ordinary (non-pipeline) model path."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable


class DistributedModel:
    """A deliberately thin Batch 0 wrapper around ``torch.nn.Module``."""

    def __init__(self, module: Any) -> None:
        self.module = module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True) -> "DistributedModel":
        self.module.train(mode)
        return self

    def eval(self) -> "DistributedModel":
        return self.train(False)

    def parameters(self, recurse: bool = True) -> Iterable[Any]:
        return self.module.parameters(recurse=recurse)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any], **kwargs: Any) -> Any:
        return self.module.load_state_dict(state, **kwargs)

    def no_sync(self):
        return getattr(self.module, "no_sync", nullcontext)()


class PipelineStage:
    """A stage module plus immutable stage metadata and schedule ownership."""

    def __init__(self, module: Any, *, stage_id: int, stage_plan: Any,
                 schedule: Any = None, pp_group: Any = None, pp_size: int = 1,
                 pp_ranks: tuple[int, ...] | None = None, num_microbatches: int = 1,
                 tp_rank: int = 0, dp_rank: int = 0) -> None:
        self.module = module
        self.stage_id = stage_id
        self.stage_plan = stage_plan
        self.schedule = schedule
        self.pp_group = pp_group
        self.pp_size = pp_size
        self.pp_ranks = pp_ranks
        self.num_microbatches = num_microbatches
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self._pipeline_backward_complete = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterable[Any]:
        return self.module.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> "PipelineStage":
        self.module.train(mode)
        return self

    def eval(self) -> "PipelineStage":
        return self.train(False)

    def no_sync(self):
        return self.module.no_sync() if hasattr(self.module, "no_sync") else nullcontext()

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> Any:
        if hasattr(self.module, "clip_grad_norm_"):
            return self.module.clip_grad_norm_(max_norm, norm_type)
        import torch
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm, norm_type=norm_type)

    def to(self, *args: Any, **kwargs: Any) -> "PipelineStage":
        self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any], **kwargs: Any) -> Any:
        return self.module.load_state_dict(state, **kwargs)

    @property
    def backward_complete(self) -> bool:
        # On the production path `schedule` is a plain STRING ("gpipe"/"1f1b") --
        # pipeline_step dispatches on it -- so consulting it for a flag is
        # misleading; the instance flag below is what actually tracks completion.
        from_schedule = (getattr(self.schedule, "_backward_complete", False)
                         if not isinstance(self.schedule, str) else False)
        return bool(self._pipeline_backward_complete or from_schedule)

    @property
    def uses_pipeline(self) -> bool:
        return self.pp_size > 1

    def pipeline_step(self, batch: Any, *, loss_fn: Any = None, scaler: Any = None,
                      accumulation_steps: int = 1) -> dict[str, Any]:
        """Execute a synchronous GPipe-style step for the current stage.

        The stage owns the complete forward/backward lifecycle, so Trainer
        must not call ``backward`` again.  The same explicit metadata path is
        used for both configured schedules; the 1F1B path uses nonblocking
        header/payload sends and a bounded activation FIFO.
        """
        import torch
        from .parallel.pp_p2p import P2PCommunicator
        from .parallel.pp_shapes import TensorSpec, split_microbatches

        if self.pp_size <= 1:
            if isinstance(batch, dict):
                values = {key: value for key, value in batch.items()
                          if key not in {"labels", "target", "targets"}}
                output = self.module(**values) if values else self.module(batch)
            elif isinstance(batch, (tuple, list)):
                output = self.module(batch[0])
            else:
                output = self.module(batch)
            if loss_fn is None:
                if isinstance(batch, dict) and "labels" in batch:
                    logits = output["logits"] if isinstance(output, dict) else output
                    labels = batch["labels"]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                else:
                    raise ValueError("pipeline loss_fn is required unless batch contains labels")
            else:
                loss = loss_fn(output, batch)
            scaled = loss / max(1, accumulation_steps)
            (scaler.scale(scaled) if scaler is not None and scaler.is_enabled() else scaled).backward()
            return {"loss": loss.detach(), "is_last_stage": True, "backward_complete": True}

        if self.schedule == "1f1b":
            return self._pipeline_step_1f1b(batch, loss_fn=loss_fn, scaler=scaler,
                                            accumulation_steps=accumulation_steps)

        communicator = P2PCommunicator(pp_rank=self.stage_id, pp_size=self.pp_size,
                                       pp_group=self.pp_group, pp_ranks=self.pp_ranks)
        batches = split_microbatches(batch, self.num_microbatches)
        # Backward divides by the accumulation window as well as the microbatch
        # count, so N accumulated steps average instead of summing.
        denom = len(batches) * max(1, accumulation_steps)

        def call(item: Any) -> Any:
            if isinstance(item, dict):
                values = {key: value for key, value in item.items() if key not in {"labels", "target", "targets"}}
                return self.module(**values) if values else self.module(item)
            if isinstance(item, (tuple, list)):
                return self.module(item[0])
            return self.module(item)

        def compute_loss(output: Any, item: Any) -> Any:
            if loss_fn is not None:
                return loss_fn(output, item)
            if isinstance(item, dict) and "labels" in item:
                logits = output["logits"] if isinstance(output, dict) else output
                labels = item["labels"]
                return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                # Same reduction as every sibling path (the pp_size<=1 branch just
                # above, the dict branch, _compute_loss, PipelineSchedule._loss).
                # This used mse_loss, so enabling PP silently switched a
                # classification objective to a regression one.
                logits = output["logits"] if isinstance(output, dict) else output
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), item[1].reshape(-1))
            raise ValueError("last pipeline stage requires loss_fn or labels/targets in the batch")

        local_inputs: list[tuple[Any, TensorSpec]] = []
        local_outputs: list[Any] = []
        losses: list[Any] = []
        if communicator.is_first:
            for index, item in enumerate(batches):
                output = call(item)
                spec = TensorSpec.from_tensor(output, stage=self.stage_id, microbatch=index, tag="forward")
                communicator.send_forward(output, spec=spec)
                local_outputs.append(output)
            for output in reversed(local_outputs):
                grad, _ = communicator.recv_backward_dynamic(device=output.device)
                torch.autograd.backward(output, grad)
        elif communicator.is_last:
            last_records: list[tuple[Any, TensorSpec, Any]] = []
            for item in batches:
                activation, spec = communicator.recv_forward_dynamic(device=next(self.parameters()).device)
                activation.retain_grad()
                output = call(activation)
                loss = compute_loss(output, item) / denom
                last_records.append((activation, spec, loss))
            for activation, spec, loss in reversed(last_records):
                (scaler.scale(loss) if scaler is not None and scaler.is_enabled() else loss).backward()
                backward_spec = TensorSpec(spec.shape, spec.dtype, spec.device, spec.stage, spec.microbatch, "backward")
                communicator.send_backward(activation.grad, spec=backward_spec)
                losses.append(loss.detach() * denom)
        else:
            for index in range(self.num_microbatches):
                activation, spec = communicator.recv_forward_dynamic(device=next(self.parameters()).device)
                output = call(activation)
                output_spec = TensorSpec.from_tensor(output, stage=self.stage_id, microbatch=index, tag="forward")
                communicator.send_forward(output, spec=output_spec)
                local_inputs.append((activation, spec))
            for activation, spec in reversed(local_inputs):
                grad, _ = communicator.recv_backward_dynamic(device=activation.device)
                torch.autograd.backward(activation, grad)
                backward_spec = TensorSpec(spec.shape, spec.dtype, spec.device, spec.stage, spec.microbatch, "backward")
                communicator.send_backward(activation.grad, spec=backward_spec)
        communicator.drain()
        self._pipeline_backward_complete = True
        loss_value = torch.stack(losses).mean() if losses else None
        return {"loss": loss_value, "is_last_stage": communicator.is_last,
                "backward_complete": True}

    def _pipeline_step_1f1b(self, batch: Any, *, loss_fn: Any = None,
                            scaler: Any = None, accumulation_steps: int = 1) -> dict[str, Any]:
        """Run non-interleaved 1F1B using nonblocking P2P sends."""
        import torch
        from .parallel.pp_p2p import P2PCommunicator
        from .parallel.pp_shapes import TensorSpec, split_microbatches

        communicator = P2PCommunicator(pp_rank=self.stage_id, pp_size=self.pp_size,
                                       pp_group=self.pp_group, pp_ranks=self.pp_ranks)
        batches = split_microbatches(batch, self.num_microbatches)
        denom = len(batches) * max(1, accumulation_steps)
        pending: list[Any] = []
        records: list[tuple[Any, TensorSpec, Any | None]] = []
        losses: list[Any] = []

        def call(item: Any) -> Any:
            if isinstance(item, dict):
                values = {key: value for key, value in item.items() if key not in {"labels", "target", "targets"}}
                return self.module(**values) if values else self.module(item)
            if isinstance(item, (tuple, list)):
                return self.module(item[0])
            return self.module(item)

        def compute_loss(output: Any, item: Any) -> Any:
            if loss_fn is not None:
                return loss_fn(output, item)
            if isinstance(item, dict) and "labels" in item:
                logits = output["logits"] if isinstance(output, dict) else output
                return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), item["labels"].reshape(-1))
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                # Same reduction as every sibling path (the pp_size<=1 branch just
                # above, the dict branch, _compute_loss, PipelineSchedule._loss).
                # This used mse_loss, so enabling PP silently switched a
                # classification objective to a regression one.
                logits = output["logits"] if isinstance(output, dict) else output
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), item[1].reshape(-1))
            raise ValueError("last pipeline stage requires loss_fn or labels/targets in the batch")

        def forward(index: int) -> None:
            if communicator.is_first:
                output = call(batches[index])
                spec = TensorSpec.from_tensor(output, stage=self.stage_id, microbatch=index, tag="forward")
                pending.append(communicator.send_forward_async(output, spec=spec))
                records.append((output, spec, None))
                return
            activation, spec = communicator.recv_forward_dynamic(device=next(self.parameters()).device)
            output = call(activation)
            loss = compute_loss(output, batches[index]) / denom if communicator.is_last else None
            if communicator.is_last:
                records.append((activation, spec, loss))
            else:
                output_spec = TensorSpec.from_tensor(output, stage=self.stage_id, microbatch=index, tag="forward")
                pending.append(communicator.send_forward_async(output, spec=output_spec))
                records.append((activation, spec, None))

        def backward(record: tuple[Any, TensorSpec, Any | None]) -> None:
            activation, spec, loss = record
            if communicator.is_first:
                grad, _ = communicator.recv_backward_dynamic(device=activation.device)
                torch.autograd.backward(activation, grad)
                return
            if communicator.is_last:
                (scaler.scale(loss) if scaler is not None and scaler.is_enabled() else loss).backward()
            else:
                grad, _ = communicator.recv_backward_dynamic(device=activation.device)
                torch.autograd.backward(activation, grad)
            backward_spec = TensorSpec(spec.shape, spec.dtype, spec.device, spec.stage, spec.microbatch, "backward")
            pending.append(communicator.send_backward_async(activation.grad, spec=backward_spec))
            if loss is not None:
                losses.append(loss.detach() * denom)

        warmup = min(self.pp_size - self.stage_id - 1, len(batches))
        for index in range(warmup):
            forward(index)
        for work in pending:
            work.wait()
        pending.clear()
        next_index = warmup
        while next_index < len(batches):
            forward(next_index)
            next_index += 1
            backward(records.pop(0))
        while records:
            backward(records.pop(0))
        for work in pending:
            work.wait()
        communicator.drain()
        self._pipeline_backward_complete = True
        return {"loss": torch.stack(losses).mean() if losses else None,
                "is_last_stage": communicator.is_last, "backward_complete": True}

    def pipeline_evaluate(self, batch: Any, *, loss_fn: Any = None) -> dict[str, Any]:
        """Forward-only PP traversal used by ``Trainer.evaluate``."""
        import torch
        from .parallel.pp_p2p import P2PCommunicator
        from .parallel.pp_shapes import TensorSpec, split_microbatches
        if self.pp_size <= 1:
            output = self.module(**batch) if isinstance(batch, dict) else self.module(*batch) if isinstance(batch, (tuple, list)) else self.module(batch)
            loss = loss_fn(output, batch) if loss_fn is not None else None
            return {"loss": loss.detach() if torch.is_tensor(loss) else None, "is_last_stage": True}
        communicator = P2PCommunicator(pp_rank=self.stage_id, pp_size=self.pp_size,
                                       pp_group=self.pp_group, pp_ranks=self.pp_ranks)
        batches = split_microbatches(batch, self.num_microbatches)
        losses: list[Any] = []
        if communicator.is_first:
            for index, item in enumerate(batches):
                # Same convention as ``call`` above: a Mapping must not be passed
                # positionally, and supervision never reaches the module.
                if isinstance(item, dict):
                    values = {key: value for key, value in item.items()
                              if key not in {"labels", "target", "targets"}}
                    output = self.module(**values) if values else self.module(item)
                elif isinstance(item, (tuple, list)):
                    output = self.module(item[0])
                else:
                    output = self.module(item)
                communicator.send_forward(output, spec=TensorSpec.from_tensor(output, stage=self.stage_id,
                                                                                microbatch=index, tag="forward"))
        elif communicator.is_last:
            for item in batches:
                activation, _ = communicator.recv_forward_dynamic(device=next(self.parameters()).device)
                output = self.module(activation)
                if loss_fn is not None:
                    losses.append(loss_fn(output, item).detach())
        else:
            for index in range(self.num_microbatches):
                activation, _ = communicator.recv_forward_dynamic(device=next(self.parameters()).device)
                output = self.module(activation)
                communicator.send_forward(output, spec=TensorSpec.from_tensor(output, stage=self.stage_id,
                                                                                microbatch=index, tag="forward"))
        communicator.drain()
        return {"loss": torch.stack(losses).mean() if losses else None,
                "is_last_stage": communicator.is_last}
