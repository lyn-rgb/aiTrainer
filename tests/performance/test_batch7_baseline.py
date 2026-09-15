"""Repeatable benchmark entry point; CUDA measurements require the target environment."""

import time

from aitrainer.profiler import Profiler


def test_synchronous_profiler_baseline():
    profiler = Profiler()
    with profiler.section("eager_baseline"):
        time.perf_counter()
    summary = profiler.summary()
    assert summary["eager_baseline"] >= 0.0
    assert summary["overlap_ratio"] == 0.0
