from aitrainer.memory import BufferKey, BufferPoolError, PinnedBufferPool


def test_pool_quota_and_reuse():
    pool = PinnedBufferPool(16, allocator=lambda key, pin: bytearray(key.nbytes))
    key = BufferKey((4,), "float32")
    value = pool.acquire(key)
    pool.release(value)
    assert pool.acquire(key) is value
    pool.release(value)
    try:
        pool.acquire(BufferKey((8,), "float32"))
    except BufferPoolError:
        return
    raise AssertionError("quota must be hard")
