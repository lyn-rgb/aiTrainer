#!/usr/bin/env python3
"""Measure forward/backward efficiency, memory and one-step correctness.

This benchmark intentionally uses a small deterministic MLP so it can run on
CPU, one CUDA device, or a single rank of a server. It is a measurement tool,
not a pass/fail claim about a particular GPU; all raw values are preserved in
the JSON output for comparison with later runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class RunMetrics:
    name: str
    device: str
    steps: int
    batch_size: int
    duration_seconds: float
    steps_per_second: float
    samples_per_second: float
    final_loss: float
    finite: bool
    cpu_max_rss_bytes: int
    cpu_current_rss_bytes: int
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    cuda_current_allocated_bytes: int | None
    cuda_current_reserved_bytes: int | None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return _rss_bytes()


def _sync(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _memory_start(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _memory_values(torch: Any, device: Any) -> tuple[int | None, int | None, int | None, int | None]:
    if getattr(device, "type", str(device)) != "cuda":
        return None, None, None, None
    return (int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device)),
            int(torch.cuda.memory_allocated(device)), int(torch.cuda.memory_reserved(device)))


def _make_model(torch: Any) -> Any:
    return torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, 32))


def _run(torch: Any, *, name: str, device: Any, steps: int, batch_size: int,
         use_offload: bool) -> RunMetrics:
    from aitrainer import FrameworkConfig, OffloadConfig, OffloadManager
    torch.manual_seed(991)
    model = _make_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = OffloadManager(FrameworkConfig(offload=OffloadConfig(enabled=True, optimizer=True)).offload) if use_offload else OffloadManager()
    manager.register_optimizer(optimizer)
    x = torch.randn(batch_size, 128, device=device)
    y = torch.randint(0, 32, (batch_size,), device=device)
    for _ in range(2):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        manager.step_optimizer(optimizer, optimizer.step)
        optimizer.zero_grad(set_to_none=True)
    _sync(torch, device)
    _memory_start(torch, device)
    started = time.perf_counter()
    final_loss = float("nan")
    finite = True
    for _ in range(steps):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        if not bool(torch.isfinite(loss).item()):
            finite = False
        loss.backward()
        manager.step_optimizer(optimizer, optimizer.step)
        optimizer.zero_grad(set_to_none=True)
        final_loss = float(loss.detach().float().item())
    _sync(torch, device)
    elapsed = time.perf_counter() - started
    peak_allocated, peak_reserved, current_allocated, current_reserved = _memory_values(torch, device)
    manager.close(optimizer)
    return RunMetrics(name, str(device), steps, batch_size, elapsed, steps / elapsed if elapsed else 0.0,
                      steps * batch_size / elapsed if elapsed else 0.0, final_loss, finite, _rss_bytes(), _current_rss_bytes(),
                      peak_allocated, peak_reserved, current_allocated, current_reserved)


def _correctness(torch: Any, device: Any) -> dict[str, Any]:
    torch.manual_seed(992)
    first, second = _make_model(torch).to(device), _make_model(torch).to(device)
    second.load_state_dict(first.state_dict())
    x = torch.randn(8, 128, device=device)
    y = torch.randint(0, 32, (8,), device=device)
    output_first, output_second = first(x), second(x)
    max_output_error = float((output_first - output_second).abs().max().item())
    loss_first = torch.nn.functional.cross_entropy(output_first, y)
    loss_second = torch.nn.functional.cross_entropy(output_second, y)
    loss_first.backward()
    loss_second.backward()
    max_gradient_error = 0.0
    for expected, actual in zip(first.parameters(), second.parameters()):
        max_gradient_error = max(max_gradient_error, float((expected.grad - actual.grad).abs().max().item()))
    torch.testing.assert_close(loss_first, loss_second, rtol=0.0, atol=0.0)
    return {"passed": True, "max_output_error": max_output_error,
            "max_gradient_error": max_gradient_error,
            "loss": float(loss_first.detach().float().item()),
            "finite": bool(torch.isfinite(loss_first).item())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.repeats < 1:
        parser.error("--steps, --batch-size and --repeats must be positive")
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch is required for benchmark_training.py: {exc}", file=sys.stderr)
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correctness = _correctness(torch, device)
    runs = []
    for repeat in range(args.repeats):
        runs.append(_run(torch, name=f"baseline#{repeat + 1}", device=device, steps=args.steps,
                         batch_size=args.batch_size, use_offload=False))
    for repeat in range(args.repeats):
        runs.append(_run(torch, name=f"optimizer_offload#{repeat + 1}", device=device, steps=args.steps,
                         batch_size=args.batch_size, use_offload=True))
    baseline_runs = runs[:args.repeats]
    offload_runs = runs[args.repeats:]
    payload = {"environment": {"torch": torch.__version__, "device": str(device),
                                "cuda_available": bool(torch.cuda.is_available()),
                                "cuda_device_count": int(torch.cuda.device_count())},
               "configuration": {"steps": args.steps, "batch_size": args.batch_size, "repeats": args.repeats},
               "correctness": correctness, "runs": [asdict(run) for run in runs],
               "comparison": {"baseline_mean_samples_per_second": statistics.mean(item.samples_per_second for item in baseline_runs),
                              "offload_mean_samples_per_second": statistics.mean(item.samples_per_second for item in offload_runs),
                              "offload_samples_per_second_delta": statistics.mean(item.samples_per_second for item in offload_runs) - statistics.mean(item.samples_per_second for item in baseline_runs),
                              "baseline_mean_peak_allocated_bytes": (None if baseline_runs[0].cuda_peak_allocated_bytes is None else
                                                                       statistics.mean(item.cuda_peak_allocated_bytes for item in baseline_runs)),
                              "offload_mean_peak_allocated_bytes": (None if offload_runs[0].cuda_peak_allocated_bytes is None else
                                                                      statistics.mean(item.cuda_peak_allocated_bytes for item in offload_runs))}}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if correctness["passed"] and all(run.finite and math.isfinite(run.final_loss) for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
