"""GPipe and non-interleaved 1F1B pipeline schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..data import model_inputs
from .pp_p2p import P2PCommunicator
from .pp_shapes import TensorSpec, normalize_loss, split_microbatches


class PipelineScheduleError(RuntimeError):
    """Raised when a schedule cannot satisfy its stage or micro-batch contract."""


@dataclass(frozen=True)
class ScheduleOutput:
    loss: Any | None
    outputs: tuple[Any, ...]
    microbatches: int
    backward_complete: bool


def _call_module(module: Any, batch: Any) -> Any:
    args, kwargs = model_inputs(batch)
    return module(*args, **kwargs)


def _extract_target(batch: Any) -> Any:
    if isinstance(batch, Mapping):
        if "labels" not in batch:
            raise PipelineScheduleError("last pipeline stage requires batch['labels'] or a custom loss adapter")
        return batch["labels"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[1]
    raise PipelineScheduleError("last pipeline stage requires a target in the batch")


class PipelineSchedule:
    """Base schedule with local and distributed execution paths."""

    def __init__(self, *, stage_module: Any, communicator: P2PCommunicator,
                 num_microbatches: int, loss_fn: Callable[[Any, Any], Any] | None = None,
                 stage_id: int | None = None) -> None:
        if num_microbatches < 1:
            raise PipelineScheduleError("num_microbatches must be positive")
        self.stage_module = stage_module
        self.communicator = communicator
        self.num_microbatches = num_microbatches
        self.loss_fn = loss_fn
        # A stage's id IS its rank.  Defaulting this to 0 independently of the
        # communicator labelled every stage's forwards "stage 0", which the
        # receiver's metadata check then rejects -- or worse, accepts at
        # pp_size == 2 where 0 happens to be a neighbour.
        if stage_id is not None and stage_id != communicator.pp_rank:
            raise PipelineScheduleError(
                f"stage_id={stage_id} disagrees with communicator.pp_rank={communicator.pp_rank}")
        self.stage_id = communicator.pp_rank if stage_id is None else stage_id
        self._backward_complete = False

    def _loss(self, output: Any, batch: Any) -> Any:
        if self.loss_fn is not None:
            return normalize_loss(self.loss_fn(output, batch))
        target = _extract_target(batch)
        import torch.nn.functional as functional
        logits = output["logits"] if isinstance(output, Mapping) and "logits" in output else output
        return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    def run(self, batch: Any) -> ScheduleOutput:
        """Run the selected schedule, using a local path for ``pp_size=1``."""
        if self.communicator.pp_size == 1:
            return self._run_local(batch)
        return self._run_distributed(batch)

    def _run_local(self, batch: Any) -> ScheduleOutput:
        batches = split_microbatches(batch, self.num_microbatches)
        outputs, losses = [], []
        for item in batches:
            output = _call_module(self.stage_module, item)
            outputs.append(output)
            raw_loss = self._loss(output, item)
            losses.append(raw_loss)
            (raw_loss / len(batches)).backward()
        self._backward_complete = True
        loss_value = sum(losses) / len(losses) if losses else None
        return ScheduleOutput(loss_value, tuple(outputs), len(batches), True)

    def _run_distributed(self, batch: Any) -> ScheduleOutput:
        raise PipelineScheduleError("use GPipeSchedule or OneFOneBSchedule for a concrete pipeline schedule")

    def _input_spec(self, tensor: Any, microbatch: int, tag: str) -> TensorSpec:
        return TensorSpec.from_tensor(tensor, stage=self.stage_id, microbatch=microbatch, tag=tag)

    @staticmethod
    def _backward_spec(spec: TensorSpec) -> TensorSpec:
        return TensorSpec(spec.shape, spec.dtype, spec.device, spec.stage, spec.microbatch, "backward")

    def _recv_backward_spec(self, spec: TensorSpec) -> TensorSpec:
        """Metadata for a gradient arriving from the DOWNSTREAM stage.

        The payload is the gradient of THIS stage's output, and the neighbour
        labels it with this stage's id (it reuses the forward spec this stage
        sent).  Reusing the forward spec on the receive side instead made an
        interior stage expect ``own_id - 1`` while its neighbour sent
        ``own_id`` -- a mismatch at every pp_size >= 3, masked at pp_size == 2.
        """
        return TensorSpec(spec.shape, spec.dtype, spec.device, self.communicator.pp_rank,
                          spec.microbatch, "backward")

    def _forward_stage0(self, item: Any, microbatch: int) -> tuple[Any, TensorSpec]:
        output = _call_module(self.stage_module, item)
        spec = self._input_spec(output, microbatch, "forward")
        self.communicator.send_forward(output, spec=spec)
        return output, spec

    def _forward_middle(self, spec: TensorSpec) -> tuple[Any, TensorSpec]:
        activation = self.communicator.recv_forward(spec)
        output = self.stage_module(activation)
        next_spec = self._input_spec(output, spec.microbatch, "forward")
        self.communicator.send_forward(output, spec=next_spec)
        return activation, next_spec

    def _forward_last(self, item: Any, spec: TensorSpec) -> tuple[Any, Any]:
        activation = self.communicator.recv_forward(spec)
        output = self.stage_module(activation)
        loss = self._loss(output, item)
        return activation, loss

    def _backward_activation(self, activation: Any, grad: Any, *, spec: TensorSpec) -> Any:
        torch = __import__("torch")
        torch.autograd.backward(activation, grad)
        return activation.grad


class GPipeSchedule(PipelineSchedule):
    """All-forward then all-backward schedule with FIFO activation storage."""

    def _run_distributed(self, batch: Any) -> ScheduleOutput:
        batches = split_microbatches(batch, self.num_microbatches) if self.communicator.is_first else None
        activations: list[Any] = []
        specs: list[TensorSpec] = []
        outputs: list[Any] = []
        if self.communicator.is_first:
            for index, item in enumerate(batches or []):
                output, spec = self._forward_stage0(item, index)
                activations.append(output)
                specs.append(spec)
        elif self.communicator.is_last:
            # Last stage receives metadata from the caller's stage plan; the
            # spec is supplied through ``input_spec`` in distributed wrappers.
            raise PipelineScheduleError("last-stage GPipe requires run_last_stage with input specs")
        else:
            raise PipelineScheduleError("middle-stage GPipe requires run_middle_stage with input specs")
        for index in range(len(activations) - 1, -1, -1):
            grad = self.communicator.recv_backward(self._recv_backward_spec(specs[index]))
            torch = __import__("torch")
            torch.autograd.backward(activations[index], grad)
        self._backward_complete = True
        return ScheduleOutput(None, tuple(outputs), len(activations), True)

    def run_last_stage(self, batches: list[Any], input_specs: list[TensorSpec]) -> ScheduleOutput:
        if not self.communicator.is_last:
            raise PipelineScheduleError("run_last_stage called on a non-last stage")
        activations, losses = [], []
        records = []
        if len(batches) != len(input_specs):
            # zip() would silently truncate: the leftover activations stay queued
            # upstream (which then blocks in recv_backward) and fewer gradients are
            # sent than were received.
            raise PipelineScheduleError(
                f"run_last_stage needs one input spec per microbatch; got "
                f"batches={len(batches)} specs={len(input_specs)}")
        for item, spec in zip(batches, input_specs):
            activation = self.communicator.recv_forward(spec)
            activation.retain_grad()
            output = self.stage_module(activation)
            loss = self._loss(output, item)
            records.append((activation, spec, loss))
            activations.append(output)
            losses.append(loss.detach())
        # Peers drain their activation FIFO in REVERSE order, so the gradients have
        # to leave descending too.  Sending inside the forward loop above shipped
        # them ascending and tripped the microbatch metadata check (needs
        # num_microbatches >= 2 to be observable).
        for activation, spec, loss in reversed(records):
            (loss / len(records)).backward()
            self.communicator.send_backward(activation.grad, spec=self._backward_spec(spec))
        self._backward_complete = True
        mean_loss = sum(losses) / len(losses) if losses else None
        return ScheduleOutput(mean_loss, tuple(activations), len(activations), True)

    def run_middle_stage(self, input_specs: list[TensorSpec]) -> ScheduleOutput:
        if self.communicator.is_first or self.communicator.is_last:
            raise PipelineScheduleError("run_middle_stage requires an interior pipeline stage")
        fifo: list[tuple[Any, TensorSpec]] = []
        for spec in input_specs:
            activation = self.communicator.recv_forward(spec)
            output = self.stage_module(activation)
            output_spec = self._input_spec(output, spec.microbatch, "forward")
            self.communicator.send_forward(output, spec=output_spec)
            fifo.append((activation, spec))
        import torch
        for activation, spec in reversed(fifo):
            grad = self.communicator.recv_backward(self._recv_backward_spec(spec))
            torch.autograd.backward(activation, grad)
            self.communicator.send_backward(activation.grad, spec=self._backward_spec(spec))
        self._backward_complete = True
        return ScheduleOutput(None, (), len(fifo), True)


class OneFOneBSchedule(PipelineSchedule):
    """Non-interleaved warmup/steady/cooldown 1F1B schedule."""

    def _run_distributed(self, batch: Any) -> ScheduleOutput:
        batches = split_microbatches(batch, self.num_microbatches) if self.communicator.is_first else None
        if self.communicator.is_last:
            raise PipelineScheduleError("last-stage 1F1B requires run_last_stage with input specs")
        if not self.communicator.is_first:
            raise PipelineScheduleError("middle-stage 1F1B requires run_middle_stage with stage input specs")
        # Stage 0 owns the full warmup/steady/cooldown FIFO; peers run their
        # matching ``run_middle_stage``/``run_last_stage`` calls.
        fifo: list[tuple[Any, TensorSpec]] = []
        outputs: list[Any] = []
        for index, item in enumerate(batches or []):
            output, spec = self._forward_stage0(item, index)
            fifo.append((output, spec))
            warmup = min(self.communicator.pp_size - 1, self.num_microbatches)
            if index >= warmup:
                activation, activation_spec = fifo.pop(0)
                grad = self.communicator.recv_backward(self._recv_backward_spec(activation_spec))
                import torch
                torch.autograd.backward(activation, grad)
                outputs.append(output)
        while fifo:
            activation, spec = fifo.pop(0)
            grad = self.communicator.recv_backward(self._recv_backward_spec(spec))
            import torch
            torch.autograd.backward(activation, grad)
        self._backward_complete = True
        return ScheduleOutput(None, tuple(outputs), len(batches or []), True)

    def run_last_stage(self, batches: list[Any], input_specs: list[TensorSpec]) -> ScheduleOutput:
        # A deterministic FIFO implementation is used for the last stage;
        # forward/backward alternation is explicit and activations are released
        # immediately after their gradient is sent upstream.
        if not self.communicator.is_last:
            raise PipelineScheduleError("run_last_stage called on a non-last stage")
        losses, outputs = [], []
        if len(batches) != len(input_specs):
            raise PipelineScheduleError(
                f"run_last_stage needs one input spec per microbatch; got "
                f"batches={len(batches)} specs={len(input_specs)}")
        for item, spec in zip(batches, input_specs):
            activation = self.communicator.recv_forward(spec)
            activation.retain_grad()
            output = self.stage_module(activation)
            loss = self._loss(output, item)
            (loss / len(batches)).backward()
            self.communicator.send_backward(activation.grad, spec=self._backward_spec(spec))
            outputs.append(output)
            losses.append(loss.detach())
        self._backward_complete = True
        mean_loss = sum(losses) / len(losses) if losses else None
        return ScheduleOutput(mean_loss, tuple(outputs), len(outputs), True)

    def run_middle_stage(self, input_specs: list[TensorSpec]) -> ScheduleOutput:
        if self.communicator.is_first or self.communicator.is_last:
            raise PipelineScheduleError("run_middle_stage requires an interior pipeline stage")
        # Mirror the verified steady-state ordering of
        # PipelineStage._pipeline_step_1f1b: hold at most warmup+1 activations,
        # then alternate one forward with one backward (oldest first), then drain
        # the cooldown.  The previous body consumed ALL inputs before any backward
        # -- that is GPipe semantics.  With num_microbatches > pp_size it
        # deadlocked (config.py mandates M >= pp_size for 1f1b), and at
        # M == pp_size it tripped the microbatch metadata check.
        import torch
        total = len(input_specs)
        if not total:
            self._backward_complete = True
            return ScheduleOutput(None, (), 0, True)
        fifo: list[tuple[Any, TensorSpec]] = []

        def forward_one(spec: TensorSpec) -> None:
            activation = self.communicator.recv_forward(spec)
            output = self.stage_module(activation)
            output_spec = self._input_spec(output, spec.microbatch, "forward")
            self.communicator.send_forward(output, spec=output_spec)
            fifo.append((activation, spec))

        def backward_one() -> None:
            activation, spec = fifo.pop(0)          # oldest first, matching the forward order
            grad = self.communicator.recv_backward(self._recv_backward_spec(spec))
            torch.autograd.backward(activation, grad)
            self.communicator.send_backward(activation.grad, spec=self._backward_spec(spec))

        warmup = min(self.communicator.pp_size - self.communicator.pp_rank - 1, total)
        for spec in input_specs[:warmup]:
            forward_one(spec)
        for spec in input_specs[warmup:]:
            forward_one(spec)
            backward_one()
        while fifo:
            backward_one()
        self._backward_complete = True
        return ScheduleOutput(None, (), total, True)

