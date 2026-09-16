"""GPipe and non-interleaved 1F1B pipeline schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..core.batching import compute_loss, model_inputs
from .pp_p2p import P2PCommunicator
from .pp_shapes import TensorSpec, normalize_loss, split_microbatches


class PipelineScheduleError(RuntimeError):
    """Raised when a schedule cannot satisfy its stage or micro-batch contract."""


def _no_grad():
    import torch
    return torch.no_grad()


def _detach(value: Any) -> Any:
    return value.detach() if hasattr(value, "detach") else None


@dataclass(frozen=True)
class ScheduleOutput:
    loss: Any | None
    outputs: tuple[Any, ...]
    microbatches: int
    backward_complete: bool


def _call_module(module: Any, batch: Any) -> Any:
    args, kwargs = model_inputs(batch)
    return module(*args, **kwargs)


class PipelineSchedule:
    """Base schedule with local and distributed execution paths."""

    def __init__(self, *, stage_module: Any, communicator: P2PCommunicator,
                 num_microbatches: int, loss_fn: Callable[[Any, Any], Any] | None = None,
                 stage_id: int | None = None, scaler: Any = None,
                 accumulation_steps: int = 1, device: Any = None) -> None:
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
        self.scaler = scaler
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.device = device
        self._backward_complete = False

    def _loss(self, output: Any, batch: Any) -> Any:
        """Cross-entropy via the one shared loss policy (see core.batching).

        This used to be a fourth implementation that knew only ``"labels"`` and
        tuple index 1, so a batch keyed ``"target"`` -- which ``core.batching``
        strips as supervision -- raised here instead.
        """
        return normalize_loss(compute_loss(output, batch, self.loss_fn))

    # ------------------------------------------------------------------ #
    # Reception mode
    #
    # A caller that already holds the neighbour's tensor specs (the protocol
    # simulator, and any explicit stage plan) passes them and gets a validated
    # static receive.  The production path cannot: the upstream shapes are only
    # known once the header lands, so it receives dynamically and uses the spec
    # the header carried.  One algorithm, two ways to learn the metadata.
    # ------------------------------------------------------------------ #
    def _device(self) -> Any:
        if self.device is not None:
            return self.device
        for parameter in self.stage_module.parameters():
            return parameter.device
        import torch
        return torch.device("cpu")

    def _recv_forward_any(self, spec: TensorSpec | None) -> tuple[Any, TensorSpec]:
        if spec is not None:
            return self.communicator.recv_forward(spec), spec
        return self.communicator.recv_forward_dynamic(device=self._device())

    def _recv_backward_any(self, spec: TensorSpec | None) -> Any:
        if spec is not None:
            return self.communicator.recv_backward(self._recv_backward_spec(spec))
        grad, _ = self.communicator.recv_backward_dynamic(device=self._device())
        return grad

    def _backward(self, loss: Any) -> None:
        """Backward with the scaler applied when AMP is active."""
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _window_denominator(self, microbatches: int) -> int:
        """Microbatch count times the accumulation window, so N steps average."""
        return max(1, microbatches) * self.accumulation_steps

    def run_stage(self, batch: Any) -> ScheduleOutput:
        """Production entry: run whichever role this stage plays."""
        batches = split_microbatches(batch, self.num_microbatches)
        if self.communicator.is_first:
            return self.run_first_stage(batches, dynamic=True)
        if self.communicator.is_last:
            return self.run_last_stage(batches)
        return self.run_middle_stage()

    def run_first_stage(self, batches: list[Any]) -> ScheduleOutput:
        raise PipelineScheduleError(
            f"{type(self).__name__} does not implement run_first_stage")

    def evaluate(self, batch: Any, *, loss_fn: Any = None) -> ScheduleOutput:
        """Forward-only traversal used by ``Trainer.evaluate``."""
        module = self.stage_module
        if self.communicator.pp_size <= 1:
            with _no_grad():
                # Kept byte-for-byte as the production path had it, including the
                # positional spread for tuple batches.
                if isinstance(batch, Mapping):
                    output = module(**batch)
                elif isinstance(batch, (tuple, list)):
                    output = module(*batch)
                else:
                    output = module(batch)
                loss = loss_fn or self.loss_fn
                value = loss(output, batch) if loss is not None else None
            return ScheduleOutput(_detach(value), (output,), 1, False)
        batches = split_microbatches(batch, self.num_microbatches)
        losses: list[Any] = []
        outputs: list[Any] = []
        with _no_grad():
            if self.communicator.is_first:
                for index, item in enumerate(batches):
                    output = _call_module(module, item)
                    spec = self._input_spec(output, index, "forward")
                    self.communicator.send_forward(output, spec=spec)
                    outputs.append(output)
            elif self.communicator.is_last:
                for item in batches:
                    activation, _spec = self._recv_forward_any(None)
                    outputs.append(module(activation))
                    loss = loss_fn or self.loss_fn
                    if loss is not None:
                        losses.append(_detach(loss(outputs[-1], item)))
            else:
                for _index in range(self.num_microbatches):
                    activation, spec = self._recv_forward_any(None)
                    output = module(activation)
                    self.communicator.send_forward(
                        output, spec=self._input_spec(output, spec.microbatch, "forward"))
                    outputs.append(output)
        self.communicator.drain()
        mean = None
        if losses:
            import torch
            mean = torch.stack([value for value in losses if value is not None]).mean()
        return ScheduleOutput(mean, tuple(outputs), len(outputs), False)

    def run(self, batch: Any) -> ScheduleOutput:
        """Run the selected schedule, using a local path for ``pp_size=1``.

        At ``pp_size > 1`` this is the production entry: the stage figures out
        its own role from the communicator rather than being told.
        """
        if self.communicator.pp_size == 1:
            return self._run_local(batch)
        return self.run_stage(batch)

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


class GPipeSchedule(PipelineSchedule):
    """All-forward then all-backward, with the activations held in a list.

    One implementation of the algorithm per role.  ``input_specs`` selects how
    the metadata is learned: supplied by a caller that already knows the
    neighbour's shapes, or read from the header when it does not.  Everything
    else -- the descending gradient order, the ``retain_grad`` on the last
    stage's activations, the window denominator -- is shared.
    """

    def run_first_stage(self, batches: list[Any], *, dynamic: bool = False) -> ScheduleOutput:
        activations: list[Any] = []
        specs: list[TensorSpec] = []
        for index, item in enumerate(batches):
            output = _call_module(self.stage_module, item)
            spec = self._input_spec(output, index, "forward")
            self.communicator.send_forward(output, spec=spec)
            activations.append(output)
            specs.append(spec)
        # Descending: every peer drains its activation FIFO in reverse order.
        for index in range(len(activations) - 1, -1, -1):
            grad = self._recv_backward_any(None if dynamic else specs[index])
            _autograd_backward(activations[index], grad)
        self.communicator.drain()
        self._backward_complete = True
        return ScheduleOutput(None, tuple(activations), len(activations), True)

    def run_middle_stage(self, input_specs: list[TensorSpec] | None = None) -> ScheduleOutput:
        """Forward through this stage and backpropagate through BOTH ends of it.

        The two lists kept here are not interchangeable, and an earlier version
        of this method treated them as if they were: it recorded ``activation``
        -- this stage's INPUT, received from the upstream stage -- and then
        backpropagated the downstream gradient through THAT.  The input's graph
        belongs to the upstream stage, in another process, so backward through it
        computes nothing here: every parameter of this stage kept ``.grad is
        None`` forever, and ``activation.grad`` came back as the incoming
        gradient unchanged, so the upstream stage received a gradient that had
        not been multiplied by this stage's Jacobian.

        Measured at pp=4 (three interior-free axes): stages 0 and 3 updated,
        stages 1 and 2 kept ``grad is None`` on all 16 parameters, loss was
        correct to 1.8e-07 and ``backward_complete`` was True.  At pp=2 there is
        no interior stage, which is why this survived -- first and last stages
        were always right, and the interior path is the one no two-stage run
        exercises.

        Backpropagating through ``output`` does both jobs at once: it fills this
        stage's parameter gradients, and it fills ``activation.grad`` with the
        input gradient to hand upstream.  The received activation is a leaf
        (``recv_forward`` allocates and marks it), so its ``.grad`` is populated
        by that call.
        """
        if self.communicator.is_first or self.communicator.is_last:
            raise PipelineScheduleError("run_middle_stage requires an interior pipeline stage")
        explicit = input_specs is not None
        total = len(input_specs) if explicit else self.num_microbatches
        activations: list[Any] = []
        outputs: list[Any] = []
        input_specs_used: list[TensorSpec] = []
        output_specs: list[TensorSpec] = []
        for index in range(total):
            activation, spec = self._recv_forward_any(input_specs[index] if explicit else None)
            output = _call_module(self.stage_module, activation)
            next_spec = self._input_spec(output, spec.microbatch if spec is not None else index, "forward")
            self.communicator.send_forward(output, spec=next_spec)
            activations.append(activation)
            outputs.append(output)
            input_specs_used.append(spec)
            output_specs.append(next_spec)
        for index in range(len(outputs) - 1, -1, -1):
            # The gradient arriving from downstream corresponds to what this
            # stage SENT, so it belongs to ``outputs[index]``, and the spec used
            # to receive it is the one the send was described with.
            grad = self._recv_backward_any(output_specs[index])
            _autograd_backward(outputs[index], grad)
            self.communicator.send_backward(activations[index].grad,
                                            spec=self._backward_spec(input_specs_used[index]))
        self.communicator.drain()
        self._backward_complete = True
        return ScheduleOutput(None, (), len(activations), True)

    def run_last_stage(self, batches: list[Any],
                       input_specs: list[TensorSpec] | None = None) -> ScheduleOutput:
        if not self.communicator.is_last:
            raise PipelineScheduleError("run_last_stage called on a non-last stage")
        explicit = input_specs is not None
        if explicit and len(batches) != len(input_specs):
            # zip() would silently truncate: the leftover activations stay queued
            # upstream (which then blocks in recv_backward) and fewer gradients are
            # sent than were received.
            raise PipelineScheduleError(
                f"run_last_stage needs one input spec per microbatch; got "
                f"batches={len(batches)} specs={len(input_specs)}")
        denom = self._window_denominator(len(batches))
        records: list[tuple[Any, TensorSpec | None, Any]] = []
        outputs: list[Any] = []
        for index, item in enumerate(batches):
            activation, spec = self._recv_forward_any(input_specs[index] if explicit else None)
            activation.retain_grad()
            output = _call_module(self.stage_module, activation)
            records.append((activation, spec, self._loss(output, item) / denom))
            outputs.append(output)
        # Peers drain their activation FIFO in REVERSE order, so the gradients
        # have to leave descending too.  Sending inside the forward loop above
        # shipped them ascending and tripped the microbatch metadata check.
        for activation, spec, loss in reversed(records):
            self._backward(loss)
            self.communicator.send_backward(activation.grad, spec=self._backward_spec(spec))
        self.communicator.drain()
        self._backward_complete = True
        # Report the RAW mean loss: each recorded value is the window-scaled one
        # actually backpropagated, so multiply the denominator back out.  Without
        # this the caller saw the loss divided by the microbatch count.
        return ScheduleOutput(_stack_mean([loss.detach() * denom for _, _, loss in records]),
                              tuple(outputs), len(outputs), True)


class OneFOneBSchedule(PipelineSchedule):
    """Non-interleaved warmup / steady-state / cooldown, with nonblocking sends.

    ``warmup = min(pp_size - pp_rank - 1, microbatches)`` is the number of
    forwards this stage may run before it must start interleaving backward
    passes.  Consuming every input before any backward is GPipe semantics: with
    ``num_microbatches > pp_size`` it deadlocked, and at
    ``num_microbatches == pp_size`` it tripped the microbatch metadata check.
    """

    def run_first_stage(self, batches: list[Any], *, dynamic: bool = False) -> ScheduleOutput:
        return self._run(batches, is_first=True, dynamic=dynamic)

    def run_middle_stage(self, input_specs: list[TensorSpec] | None = None) -> ScheduleOutput:
        if self.communicator.is_first or self.communicator.is_last:
            raise PipelineScheduleError("run_middle_stage requires an interior pipeline stage")
        return self._run(None, is_first=False, input_specs=input_specs)

    def run_last_stage(self, batches: list[Any],
                       input_specs: list[TensorSpec] | None = None) -> ScheduleOutput:
        if not self.communicator.is_last:
            raise PipelineScheduleError("run_last_stage called on a non-last stage")
        explicit = input_specs is not None
        if explicit and len(batches) != len(input_specs):
            raise PipelineScheduleError(
                f"run_last_stage needs one input spec per microbatch; got "
                f"batches={len(batches)} specs={len(input_specs)}")
        return self._run(batches, is_first=False, input_specs=input_specs)

    def _run(self, batches: list[Any] | None, *, is_first: bool, dynamic: bool = False,
             input_specs: list[TensorSpec] | None = None) -> ScheduleOutput:
        explicit = input_specs is not None
        count = len(batches) if batches is not None else (
            len(input_specs) if explicit else self.num_microbatches)
        if not count:
            self._backward_complete = True
            return ScheduleOutput(None, (), 0, True)
        denom = self._window_denominator(count)
        pending: list[Any] = []
        # (target to backpropagate through, tensor whose .grad goes upstream,
        #  spec to RECEIVE the downstream gradient with, spec to SEND ours with,
        #  loss).  The first two are different tensors for an interior stage, and
        #  conflating them was the defect: see ``backward`` below.
        records: list[tuple[Any, Any, TensorSpec | None, TensorSpec | None, Any | None]] = []
        outputs: list[Any] = []
        losses: list[Any] = []

        def forward(index: int) -> None:
            if is_first:
                output = _call_module(self.stage_module, batches[index])
                spec = self._input_spec(output, index, "forward")
                pending.append(self.communicator.send_forward_async(output, spec=spec))
                records.append((output, output, spec, None, None))
                return
            activation, spec = self._recv_forward_any(input_specs[index] if explicit else None)
            output = _call_module(self.stage_module, activation)
            if self.communicator.is_last:
                records.append((output, activation, None, spec,
                                self._loss(output, batches[index]) / denom))
            else:
                next_spec = self._input_spec(output, spec.microbatch if spec is not None else index,
                                             "forward")
                pending.append(self.communicator.send_forward_async(output, spec=next_spec))
                records.append((output, activation, next_spec, spec, None))
            outputs.append(output)

        def backward(record: tuple[Any, Any, TensorSpec | None, TensorSpec | None, Any | None]) -> None:
            target, handoff, recv_spec, send_spec, loss = record
            if is_first:
                # The first stage has no upstream peer, so it receives its
                # gradient and stops -- it must NOT try to send one.  (Falling
                # through also reads ``.grad`` off its own module output, which is
                # not a leaf and has no gradient populated.)
                #
                # For this stage ``target`` IS its output, which is what the
                # downstream gradient corresponds to.
                _autograd_backward(target, self._recv_backward_any(None if dynamic else recv_spec))
                return
            if self.communicator.is_last:
                # Backward from the loss populates ``handoff``'s gradient (the
                # received activation, a leaf) on the way to this stage's
                # parameters, so no separate call is needed here.
                self._backward(loss)
            else:
                # Through the OUTPUT, not the received activation.  The gradient
                # from downstream corresponds to what this stage SENT; the
                # activation's graph belongs to the upstream stage, in another
                # process, so backward through it fills in nothing here -- it left
                # every interior parameter at ``grad is None`` for the whole run
                # while the loss stayed correct.  Going through the output fills
                # this stage's parameters AND ``handoff.grad``, which is the input
                # gradient to pass upstream.
                _autograd_backward(target, self._recv_backward_any(None if dynamic else recv_spec))
            pending.append(self.communicator.send_backward_async(
                handoff.grad, spec=self._backward_spec(send_spec)))
            if loss is not None:
                losses.append(loss.detach() * denom)

        warmup = min(self.communicator.pp_size - self.communicator.pp_rank - 1, count)
        for index in range(warmup):
            forward(index)
        for work in pending:
            work.wait()
        pending.clear()
        index = warmup
        while index < count:
            forward(index)
            index += 1
            backward(records.pop(0))
        while records:
            backward(records.pop(0))
        for work in pending:
            work.wait()
        self.communicator.drain()
        self._backward_complete = True
        return ScheduleOutput(_stack_mean(losses), tuple(outputs), count, True)


def _autograd_backward(tensor: Any, grad: Any) -> None:
    import torch
    torch.autograd.backward(tensor, grad)


def _stack_mean(values: list[Any]) -> Any:
    if not values:
        return None
    import torch
    return torch.stack(values).mean()


def build_schedule(name: str, **kwargs: Any) -> PipelineSchedule:
    """Resolve a schedule name to its implementation -- the only such mapping.

    ``config.parallel.pp_schedule`` is a string and ``PipelineStage`` stores it
    verbatim, while the classes take objects; resolving in one place is what
    stops the two conventions from drifting apart again.
    """
    if name in {"gpipe", ""}:
        return GPipeSchedule(**kwargs)
    if name == "1f1b":
        return OneFOneBSchedule(**kwargs)
    raise PipelineScheduleError(f"unknown pipeline schedule {name!r}")
