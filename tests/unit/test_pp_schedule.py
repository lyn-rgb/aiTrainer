import pytest

torch = pytest.importorskip("torch")

from aitrainer.parallel.pp_p2p import P2PCommunicator
from aitrainer.parallel.pp_schedule import GPipeSchedule, OneFOneBSchedule


@pytest.mark.parametrize("schedule_type", [GPipeSchedule, OneFOneBSchedule])
def test_single_stage_schedule_completes_backward(schedule_type):
    module = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    schedule = schedule_type(stage_module=module, communicator=P2PCommunicator(pp_rank=0, pp_size=1),
                             num_microbatches=2,
                             loss_fn=lambda output, batch: torch.nn.functional.mse_loss(output, batch[1]))
    batch = (torch.randn(4, 4), torch.randn(4, 2))
    result = schedule.run(batch)
    optimizer.step()
    optimizer.zero_grad()
    assert result.backward_complete and result.microbatches == 2
