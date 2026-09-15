from aitrainer.runtime import Runtime


def test_runtime_is_single_process_and_closes():
    with Runtime(device="cpu") as runtime:
        assert runtime.rank == 0
        assert runtime.world_size == 1
