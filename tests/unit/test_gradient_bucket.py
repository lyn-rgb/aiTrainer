from aitrainer.overlap import GradientBucket


def test_gradient_bucket_flushes_explicitly():
    bucket = GradientBucket("g", 4)
    assert bucket.add(bytearray(2))
    assert bucket.add(bytearray(2))
    assert bucket.ready
    assert len(bucket.flush()) == 2
    assert not bucket.ready
