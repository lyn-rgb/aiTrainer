"""Multi-rank pipeline protocol, simulated in-process (no rendezvous needed).

The schedule classes talk to a ``P2PCommunicator``, so a fake one backed by
message queues lets the real stage code run for every rank.  The fake enforces
the SAME metadata check the real header does (stage + microbatch must match what
the receiver expects), so ordering and labelling defects surface here instead of
only on a multi-GPU host.

Covered regressions:
  * GPipe sent its gradients in ascending microbatch order while every receiver
    drained its activation FIFO in reverse -> metadata mismatch for M >= 2.
  * An interior stage validated the incoming gradient against its own INPUT spec
    (``own_id - 1``) while the neighbour labelled it ``own_id`` -> mismatch at
    every pp_size >= 3, silently agreeing at pp_size == 2.
  * 1F1B's interior stage consumed ALL inputs before any backward (GPipe
    semantics), which deadlocked whenever num_microbatches > pp_size while
    config.py mandates M >= pp_size for 1f1b.
"""

from __future__ import annotations

import queue
import threading

import pytest

torch = pytest.importorskip("torch")

from aitrainer.parallel.pp_schedule import GPipeSchedule, OneFOneBSchedule
from aitrainer.parallel.pp_shapes import TensorSpec, split_microbatches

SHAPE = (4, 4)
DTYPE = "float32"
TIMEOUT = 5.0


class _Hub:
    """Message queues keyed by (direction, sender, receiver)."""

    def __init__(self) -> None:
        self._queues: dict[tuple[str, int, int], queue.Queue] = {}
        self.mismatches: list[str] = []
        self.forwards = 0
        self.backwards = 0
        # microbatch arrival order per (sender, receiver), to pin gradient ordering
        self.sent_order: dict[tuple[int, int], list[int]] = {}

    def queue_for(self, direction: str, sender: int, receiver: int) -> queue.Queue:
        return self._queues.setdefault((direction, sender, receiver), queue.Queue())

    def check(self, direction: str, expected: TensorSpec, got: TensorSpec) -> None:
        if got is None:
            self.mismatches.append(f"{direction}: no metadata arrived")
            return
        if (got.stage, got.microbatch) != (expected.stage, expected.microbatch):
            self.mismatches.append(
                f"{direction}: got stage={got.stage} mb={got.microbatch}, "
                f"expected stage={expected.stage} mb={expected.microbatch}")
        if direction == "forward":
            self.forwards += 1
        else:
            self.backwards += 1


class _FakeCommunicator:
    """Same surface the schedules use, with blocking queues and a timeout."""

    def __init__(self, pp_rank: int, pp_size: int, hub: _Hub) -> None:
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.hub = hub

    @property
    def is_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def send_forward(self, tensor, *, spec=None) -> None:
        self.hub.queue_for("forward", self.pp_rank, self.pp_rank + 1).put((tensor, spec))

    def recv_forward(self, spec=None):
        tensor, got = self.hub.queue_for("forward", self.pp_rank - 1, self.pp_rank).get(timeout=TIMEOUT)
        self.hub.check("forward", spec, got)
        # A real P2P receive allocates a fresh buffer and marks it a leaf
        # (P2PCommunicator.recv_forward ends with ``requires_grad_(True)``).
        # Handing the sender's tensor straight through would put two stages on ONE
        # autograd graph, and the downstream backward would then free the upstream
        # graph -- an artefact of the fake, not of the schedule.
        return tensor.detach().clone().requires_grad_(True)

    def send_backward(self, grad, *, spec=None) -> None:
        self.hub.sent_order.setdefault((self.pp_rank, self.pp_rank - 1), []).append(spec.microbatch)
        self.hub.queue_for("backward", self.pp_rank, self.pp_rank - 1).put((grad, spec))

    def recv_backward(self, spec=None):
        grad, got = self.hub.queue_for("backward", self.pp_rank + 1, self.pp_rank).get(timeout=TIMEOUT)
        self.hub.check("backward", spec, got)
        return grad.detach().clone()      # received into a fresh buffer, as the real path does

    def drain(self) -> None:
        return None


def _module() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4))


def _run_all(schedule_type, pp_size: int, microbatches: int):
    """Run every stage concurrently; a protocol deadlock surfaces as a timeout."""
    hub = _Hub()
    # One (inputs, targets) pair that split_microbatches divides into M chunks,
    # matching the convention the examples and unit tests use.
    batch = (torch.randn(microbatches, *SHAPE), torch.randn(microbatches, *SHAPE))
    chunks = split_microbatches(batch, microbatches)
    chunk_shape = tuple(chunks[0][0].shape)
    specs_from = {
        rank: [TensorSpec(chunk_shape, DTYPE, "cpu", rank, index, "forward")
               for index in range(microbatches)]
        for rank in range(pp_size - 1)
    }
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def stage(rank: int) -> None:
        try:
            communicator = _FakeCommunicator(rank, pp_size, hub)
            schedule = schedule_type(stage_module=_module(), communicator=communicator,
                                     num_microbatches=microbatches, stage_id=rank,
                                     loss_fn=lambda output, item: torch.nn.functional.mse_loss(
                                         output, item[1]))
            if rank == 0:
                results[rank] = schedule.run(batch)
            elif rank == pp_size - 1:
                results[rank] = schedule.run_last_stage(chunks, specs_from[pp_size - 2])
            else:
                results[rank] = schedule.run_middle_stage(specs_from[rank - 1])
        except BaseException as exc:                      # noqa: BLE001 - reported below
            errors[rank] = exc

    threads = [threading.Thread(target=stage, args=(rank,), daemon=True) for rank in range(pp_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT * 2)
    return hub, results, errors, [thread.is_alive() for thread in threads]


def _assert_clean(hub, results, errors, alive, pp_size):
    stuck = [rank for rank, still_running in enumerate(alive) if still_running]
    assert not stuck, f"stage(s) {stuck} never finished -- protocol deadlock"
    assert not errors, f"stage raised: {errors}"
    assert not hub.mismatches, f"metadata mismatches: {hub.mismatches}"
    assert len(results) == pp_size, f"only ranks {sorted(results)} produced a result"


@pytest.mark.parametrize("pp_size,microbatches", [(2, 2), (3, 2), (3, 4), (4, 4), (2, 4)])
def test_gpipe_protocol_is_consistent(pp_size, microbatches):
    hub, results, errors, alive = _run_all(GPipeSchedule, pp_size, microbatches)
    _assert_clean(hub, results, errors, alive, pp_size)
    # every rank exchanged exactly M activations and M gradients
    assert hub.forwards == microbatches * (pp_size - 1)
    assert hub.backwards == microbatches * (pp_size - 1)
    for rank in range(pp_size):
        assert results[rank].microbatches == microbatches


@pytest.mark.parametrize("pp_size,microbatches", [(2, 2), (3, 3), (3, 4), (4, 4), (2, 4)])
def test_1f1b_protocol_is_consistent(pp_size, microbatches):
    """M > pp_size is the documented 1f1b regime and used to deadlock the interior."""
    hub, results, errors, alive = _run_all(OneFOneBSchedule, pp_size, microbatches)
    _assert_clean(hub, results, errors, alive, pp_size)
    assert hub.forwards == microbatches * (pp_size - 1)
    assert hub.backwards == microbatches * (pp_size - 1)


@pytest.mark.parametrize("pp_size,microbatches", [(2, 3), (3, 4)])
def test_gpipe_sends_gradients_in_descending_microbatch_order(pp_size, microbatches):
    """Receivers drain their activation FIFO in reverse; senders must match."""
    hub, results, errors, alive = _run_all(GPipeSchedule, pp_size, microbatches)
    _assert_clean(hub, results, errors, alive, pp_size)
    assert hub.sent_order, "no gradients were sent"
    for (sender, receiver), order in hub.sent_order.items():
        assert order == sorted(order, reverse=True), f"{sender}->{receiver} sent {order}"


@pytest.mark.parametrize("pp_size,microbatches", [(2, 3), (3, 4)])
def test_1f1b_sends_gradients_in_ascending_microbatch_order(pp_size, microbatches):
    """1F1B drains its FIFO oldest-first, so gradients leave ascending."""
    hub, results, errors, alive = _run_all(OneFOneBSchedule, pp_size, microbatches)
    _assert_clean(hub, results, errors, alive, pp_size)
    assert hub.sent_order, "no gradients were sent"
    for (sender, receiver), order in hub.sent_order.items():
        assert order == sorted(order), f"{sender}->{receiver} sent {order}"


def test_the_harness_detects_a_mislabelled_gradient():
    """Negative control: without this, a green run proves nothing.

    The pre-fix code validated an incoming gradient against the stage's own
    FORWARD spec (``own_id - 1``); the neighbour labels it ``own_id``.
    """
    def receive_with(spec_builder, queue_put):
        hub = _Hub()
        communicator = _FakeCommunicator(pp_rank=1, pp_size=3, hub=hub)
        schedule = GPipeSchedule(stage_module=_module(), communicator=communicator,
                                 num_microbatches=1, loss_fn=lambda output, item: output.sum())
        forward_spec = TensorSpec(SHAPE, DTYPE, "cpu", 0, 0, "forward")
        queue_put(hub)
        communicator.recv_backward(spec_builder(schedule, forward_spec))
        return hub.mismatches

    def neighbour_gradient(hub):
        # stage 2 reuses the spec stage 1 sent forward, so it labels the gradient 1
        hub.queue_for("backward", 2, 1).put(
            (torch.zeros(*SHAPE), TensorSpec(SHAPE, DTYPE, "cpu", 1, 0, "backward")))

    fixed = receive_with(lambda s, spec: s._recv_backward_spec(spec), neighbour_gradient)
    assert not fixed, "the fixed expectation must accept the neighbour's label"

    pre_fix = receive_with(lambda s, spec: s._backward_spec(spec), neighbour_gradient)
    assert pre_fix, "the pre-fix expectation must be rejected -- otherwise the check is inert"


def test_a_stage_id_disagreeing_with_the_communicator_is_rejected():
    """The id used to default to 0 independently of the rank, mislabelling forwards."""
    hub = _Hub()
    communicator = _FakeCommunicator(pp_rank=2, pp_size=3, hub=hub)
    with pytest.raises(Exception, match="stage_id"):
        GPipeSchedule(stage_module=_module(), communicator=communicator, num_microbatches=1,
                      stage_id=0, loss_fn=lambda output, item: output.sum())
    # omitting it now derives the correct value from the rank
    schedule = GPipeSchedule(stage_module=_module(), communicator=communicator,
                             num_microbatches=1, loss_fn=lambda output, item: output.sum())
    assert schedule.stage_id == 2


def test_interior_receiver_accepts_the_neighbour_label():
    """The interior stage must expect its OWN stage id on an incoming gradient.

    Reusing the forward spec made it expect ``own_id - 1`` while the neighbour
    (which reuses the spec this stage sent forward) labelled it ``own_id``.
    """
    hub = _Hub()
    communicator = _FakeCommunicator(pp_rank=1, pp_size=3, hub=hub)
    schedule = GPipeSchedule(stage_module=_module(), communicator=communicator,
                             num_microbatches=1,
                             loss_fn=lambda output, item: output.sum())
    input_spec = TensorSpec(SHAPE, DTYPE, "cpu", 0, 0, "forward")     # from stage 0
    expected = schedule._recv_backward_spec(input_spec)
    assert expected.stage == 1 and expected.microbatch == 0

    # stage 2 reuses the spec stage 1 sent forward, so it labels the gradient 1
    sent = TensorSpec(SHAPE, DTYPE, "cpu", 1, 0, "forward")
    hub.queue_for("backward", 2, 1).put((torch.zeros(*SHAPE), sent))
    communicator.recv_backward(expected)                              # must not mismatch
    assert not hub.mismatches
