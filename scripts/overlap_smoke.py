#!/usr/bin/env python3
"""Dependency-free contract smoke for the Batch 9 overlap runtime.

This is intentionally separate from CUDA/NCCL benchmarks: it validates state,
cleanup, bounded buffers, trace fallback and the experimental two-microbatch
contract on every workstation.  The distributed matrix remains responsible for
real communication and GPU measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run() -> dict[str, object]:
    from aitrainer.core.lifecycle import AsyncOp, AsyncState, ExecutionScheduler
    from aitrainer.overlap import (GradientBucket, MicrobatchInterleaveScheduler,
                                   ParameterPrefetchCoordinator, PingPongBuffer)

    op = AsyncOp("smoke", _wait_fn=lambda _: "ready")
    scheduler = ExecutionScheduler()
    scheduler.register(op)
    assert op.wait() == "ready" and op.state is AsyncState.WAITED
    scheduler.drain()
    assert op.state is AsyncState.RELEASED

    bucket = GradientBucket("smoke", 4)
    assert bucket.add(b"ab") and bucket.add(b"cd") and bucket.is_ready
    bucket.flush()

    buffers = PingPongBuffer((bytearray(4), bytearray(4)))
    lease = buffers.acquire("smoke")
    buffers.release(lease)

    coordinator = ParameterPrefetchCoordinator(fetch_fn=lambda name: name)
    coordinator.record("module", ("parameter",))
    assert coordinator.finalize()
    fetch = coordinator.fetch("parameter")
    assert fetch is not None and fetch.wait() == "parameter"
    coordinator.release("parameter")

    interleave = MicrobatchInterleaveScheduler(tp_size=2)
    result = interleave.run([1, 2], forward_fn=lambda value: value + 1,
                            backward_fn=lambda _activation, _loss: None)
    assert result.microbatches == 2
    return {"status": "passed", "checks": ["async_state", "scheduler_drain", "gradient_bucket",
                                              "ping_pong_generation", "parameter_trace", "microbatch_interleave"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        payload = run()
    except Exception as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
