#!/usr/bin/env python3
"""Run the requested FSDP parallel-combination validation matrix.

The matrix deliberately distinguishes a capability validation failure from a
runtime failure.  All four requested configurations are validated through the
public startup validator.  The FSDP entry additionally runs the real
multi-rank benchmark; combined entries use the same composition API and are
reported with their concrete runtime result when the environment supports it.

Run with, for example::

    torchrun --standalone --nproc_per_node=2 scripts/parallel_matrix.py \
        --output artifacts/parallel_matrix.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    description: str
    dp_size: int
    tp_size: int
    pp_size: int
    sp_backend: str
    pp_schedule: str
    num_microbatches: int
    validation_world_size: int


def _world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _specs(world_size: int) -> list[MatrixSpec]:
    # SP is intentionally coupled to a TP group by ParallelConfig.  The
    # fsdp_sp entry exercises Megatron-style sequence sharding, while the
    # fsdp_sp_tp entry exercises the Ulysses/TP path.
    sp_size = 2 if world_size < 2 else 2
    sp_dp = world_size // sp_size if world_size >= 2 and world_size % sp_size == 0 else 1
    pp_dp = world_size // 4 if world_size >= 4 and world_size % 4 == 0 else 1
    pp_world = pp_dp * 2 * 2
    return [
        MatrixSpec("fsdp", "FSDP FULL_SHARD", world_size, 1, 1, "none", "none", 1, world_size),
        MatrixSpec("fsdp_sp", "FSDP + sequence parallel", sp_dp, sp_size, 1, "megatron", "none", 1, sp_dp * sp_size),
        MatrixSpec("fsdp_sp_tp", "FSDP + sequence parallel + tensor parallel", sp_dp, sp_size, 1,
                   "ulysses", "none", 1, sp_dp * sp_size),
        # Use a four-rank structural validation for this entry so a two-rank
        # smoke job still checks the combination rather than failing merely
        # because PP needs two independent axes.
        MatrixSpec("fsdp_sp_tp_pp", "FSDP + SP + TP + PP", pp_dp, 2, 2, "ulysses", "gpipe", 2, pp_world),
    ]


def _config_for(spec: MatrixSpec) -> Any:
    from aitrainer import FSDPConfig, FrameworkConfig, ParallelConfig

    return FrameworkConfig(
        parallel=ParallelConfig(dp_size=spec.dp_size, tp_size=spec.tp_size,
                                pp_size=spec.pp_size, sp_backend=spec.sp_backend,
                                pp_schedule=spec.pp_schedule,
                                num_microbatches=spec.num_microbatches),
        fsdp=FSDPConfig(enabled=True),
    )


def _validate_entry(spec: MatrixSpec) -> dict[str, Any]:
    from aitrainer import UnsupportedCombinationError, validate

    config = _config_for(spec)
    try:
        validate(config, world_size=spec.validation_world_size)
    except UnsupportedCombinationError as exc:
        return {"name": spec.name, "description": spec.description, "status": "unsupported",
                "expected_status": "supported",
                "config": config.to_dict(), "validation_world_size": spec.validation_world_size,
                "reason": str(exc)}
    except Exception as exc:
        return {"name": spec.name, "description": spec.description, "status": "invalid",
                "expected_status": "supported",
                "config": config.to_dict(), "validation_world_size": spec.validation_world_size,
                "reason": f"{type(exc).__name__}: {exc}"}
    return {"name": spec.name, "description": spec.description, "status": "validated",
            "expected_status": "supported",
            "config": config.to_dict(), "validation_world_size": spec.validation_world_size}


def _sync(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory(torch: Any, device: Any) -> dict[str, int | None]:
    if getattr(device, "type", str(device)) != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "current_allocated_bytes": None, "current_reserved_bytes": None}
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device))}


def _overlap_probe(torch: Any, dist: Any, device: Any, *, steps: int = 4) -> dict[str, Any]:
    """Measure an async collective issued before independent matrix work.

    The result is an instrumentation probe, not a claim that native FSDP
    overlaps its internal collectives.  ``estimated_overlap`` is the amount
    saved relative to a measured sequential all-reduce-plus-compute baseline.
    """
    if dist.get_world_size() < 2:
        return {"status": "blocked", "reason": "overlap requires at least two ranks"}
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)
    value = torch.ones(1 << 20, device=device)
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    dist.barrier()
    comm_samples: list[float] = []
    compute_samples: list[float] = []
    async_samples: list[float] = []
    for _ in range(max(1, steps)):
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(value)
        _sync(torch, device)
        comm_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        _ = left @ right
        _sync(torch, device)
        compute_samples.append(time.perf_counter() - started)
        dist.barrier()
        started = time.perf_counter()
        handle = dist.all_reduce(value, async_op=True)
        _ = left @ right
        _sync(torch, device)
        handle.wait()
        _sync(torch, device)
        async_samples.append(time.perf_counter() - started)
    sequential = statistics.median(comm_samples) + statistics.median(compute_samples)
    asynchronous = statistics.median(async_samples)
    estimated = max(0.0, sequential - asynchronous)
    return {"status": "passed", "method": "async_all_reduce_with_independent_matmul",
            "backend": dist.get_backend(), "device": str(device),
            "sequential_seconds": sequential, "async_seconds": asynchronous,
            "estimated_overlap_seconds": estimated,
            "estimated_overlap_ratio": estimated / sequential if sequential else 0.0,
            "communication_seconds": statistics.median(comm_samples),
            "compute_seconds": statistics.median(compute_samples),
            "samples": len(async_samples)}


def _run_fsdp(torch: Any, dist: Any, device: Any, *, steps: int, batch_size: int) -> dict[str, Any]:
    from aitrainer import DeviceMeshManager, FSDPConfig, Runtime, wrap_fsdp

    if not dist.is_initialized():
        return {"status": "blocked", "reason": "FSDP requires an initialized multi-rank process group"}
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    runtime = Runtime(device=str(device), init_process_group=False, seed=123)
    torch.manual_seed(1234)
    base = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 8))
    reference = deepcopy(base).to(device)
    candidate = deepcopy(base).to(device)
    mesh = DeviceMeshManager(dp_size=world_size, tp_size=1, pp_size=1,
                             world_size=world_size, device_type=device.type)
    fsdp_config = FSDPConfig(enabled=True)
    wrapped = wrap_fsdp(candidate, runtime=runtime, mesh=mesh, config=fsdp_config)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.05)
    torch.manual_seed(5678)
    inputs = torch.randn(batch_size, 64, device=device)
    targets = torch.randn(batch_size, 8, device=device)
    expected_output = reference(inputs)
    expected_loss = torch.nn.functional.mse_loss(expected_output, targets)
    expected_loss.backward()
    ref_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    candidate_output = wrapped(inputs)
    candidate_loss = torch.nn.functional.mse_loss(candidate_output, targets)
    output_error = float((candidate_output.detach() - expected_output.detach()).abs().max().item())
    loss_error = float((candidate_loss.detach() - expected_loss.detach()).abs().item())
    candidate_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with wrapped.full_state_dict_context():
        candidate_state = wrapped.state_dict()
    state_error = 0.0
    if rank == 0:
        expected_state = reference.state_dict()
        for key, value in expected_state.items():
            state_error = max(state_error, float((candidate_state[key].cpu() - value.cpu()).abs().max().item()))
    error_tensor = torch.tensor([output_error, state_error], dtype=torch.float64, device=device)
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    output_error, state_error = (float(error_tensor[0].item()), float(error_tensor[1].item()))
    # Warm-up before collecting timing and peak allocator statistics.
    for _ in range(2):
        out = wrapped(inputs)
        loss = torch.nn.functional.mse_loss(out, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(steps):
        out = wrapped(inputs)
        loss = torch.nn.functional.mse_loss(out, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final_loss = float(loss.detach().float().item())
    _sync(torch, device)
    elapsed = time.perf_counter() - started
    local_memory = _cuda_memory(torch, device)
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, {"rank": rank, **local_memory, "elapsed_seconds": elapsed})
    peak = [item["peak_allocated_bytes"] for item in gathered if item and item["peak_allocated_bytes"] is not None]
    reserved = [item["peak_reserved_bytes"] for item in gathered if item and item["peak_reserved_bytes"] is not None]
    return {"status": "passed" if output_error <= 1e-5 and loss_error <= 1e-6 and state_error <= 1e-5 and math.isfinite(final_loss) else "failed",
            "world_size": world_size, "device": str(device), "steps": steps, "batch_size": batch_size,
            "elapsed_seconds": elapsed, "steps_per_second": steps / elapsed if elapsed else 0.0,
            "global_samples_per_second": steps * batch_size * world_size / elapsed if elapsed else 0.0,
            "final_loss": final_loss, "output_max_abs_error": output_error,
            "loss_abs_error": loss_error,
            "parameter_max_abs_error": state_error, "finite": math.isfinite(final_loss),
            "per_rank_memory": gathered, "max_peak_allocated_bytes": max(peak) if peak else None,
            "max_peak_reserved_bytes": max(reserved) if reserved else None}


def _run_composed(torch: Any, dist: Any, device: Any, spec: MatrixSpec, *, steps: int,
                  batch_size: int) -> dict[str, Any]:
    """Exercise PP -> TP/SP -> FSDP construction and one training step."""
    from aitrainer import (DeviceMeshManager, FSDPConfig, FrameworkConfig, ParallelConfig,
                           Runtime, SequenceParallelLayerNorm, parallelize)
    if dist.get_world_size() != spec.validation_world_size:
        return {"status": "blocked", "reason":
                f"matrix entry requires world_size={spec.validation_world_size}, "
                f"launcher has {dist.get_world_size()}"}

    class ProjectionBlock(torch.nn.Module):
        def __init__(self, group: Any, use_sp: bool) -> None:
            super().__init__()
            self.norm = (SequenceParallelLayerNorm(16, process_group=group, gather_output=True)
                         if use_sp else torch.nn.LayerNorm(16))
            self.q_proj = torch.nn.Linear(16, 16)
            self.o_proj = torch.nn.Linear(16, 16)

        def forward(self, value: Any) -> Any:
            value = self.norm(value)
            return self.o_proj(torch.tanh(self.q_proj(value)))

    world_size = dist.get_world_size()
    mesh = DeviceMeshManager(pp_size=spec.pp_size, dp_size=spec.dp_size, tp_size=spec.tp_size,
                             world_size=world_size, device_type=device.type)
    torch.manual_seed(2201)
    model = torch.nn.Sequential(*(ProjectionBlock(mesh.tensor_parallel_group, spec.sp_backend != "none")
                                  for _ in range(max(2, spec.pp_size))))
    config = FrameworkConfig(
        parallel=ParallelConfig(dp_size=spec.dp_size, tp_size=spec.tp_size, pp_size=spec.pp_size,
                                sp_backend=spec.sp_backend, pp_schedule=spec.pp_schedule,
                                num_microbatches=spec.num_microbatches),
        fsdp=FSDPConfig(enabled=True),
    )
    runtime = Runtime(device=str(device), init_process_group=False, seed=2201)
    parallel_model = parallelize(model, config=config, runtime=runtime, mesh=mesh)
    optimizer = torch.optim.SGD(parallel_model.parameters(), lr=0.01)
    torch.manual_seed(2202)
    inputs = torch.randn(batch_size, 4, 16, device=device)
    targets = torch.randn(batch_size, 4, 16, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()
    if getattr(parallel_model, "uses_pipeline", False):
        loss_fn = lambda output, batch: torch.nn.functional.mse_loss(output, batch[1])
        result = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            result = parallel_model.pipeline_step((inputs, targets), loss_fn=loss_fn)
            optimizer.step()
        loss_value = result.get("loss") if result is not None else None
        local_loss = float(loss_value.item()) if torch.is_tensor(loss_value) else 0.0
    else:
        local_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = parallel_model(inputs)
            loss = torch.nn.functional.mse_loss(output, targets)
            loss.backward()
            optimizer.step()
            local_loss = float(loss.detach().item())
    _sync(torch, device)
    elapsed = time.perf_counter() - started
    local_memory = _cuda_memory(torch, device)
    per_rank: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(per_rank, {"rank": dist.get_rank(), **local_memory})
    peaks = [item["peak_allocated_bytes"] for item in per_rank
             if item and item["peak_allocated_bytes"] is not None]
    finite = math.isfinite(local_loss)
    finite_tensor = torch.tensor([1 if finite else 0], device=device, dtype=torch.int64)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    return {"status": "passed" if int(finite_tensor.item()) else "failed",
            "name": spec.name, "world_size": world_size, "loss": local_loss,
            "elapsed_seconds": elapsed, "steps_per_second": steps / elapsed if elapsed else 0.0,
            "global_samples_per_second": steps * batch_size * world_size / elapsed if elapsed else 0.0,
            "per_rank_memory": per_rank,
            "max_peak_allocated_bytes": max(peaks) if peaks else None,
            "finite": bool(finite_tensor.item()), "topology": {
                "dp_size": spec.dp_size, "tp_size": spec.tp_size, "pp_size": spec.pp_size,
                "sp_backend": spec.sp_backend}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--config", choices=("all", "fsdp"), default="all")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    world_size = _world_size()
    rank = _rank()
    matrix = [_validate_entry(spec) for spec in _specs(world_size)]
    selected = {"fsdp"} if args.config == "fsdp" else {item["name"] for item in matrix}
    fsdp_run: dict[str, Any] | None = None
    overlap_run: dict[str, Any] | None = None
    composition_runs: dict[str, dict[str, Any]] = {}
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        fsdp_run = {"status": "blocked", "reason": f"PyTorch is unavailable: {exc}"}
        overlap_run = {"status": "blocked", "reason": "PyTorch is unavailable"}
    else:
        if "fsdp" in selected and world_size >= 2:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
            if not dist.is_initialized():
                backend = "nccl" if device.type == "cuda" else "gloo"
                dist.init_process_group(backend=backend)
            try:
                fsdp_run = _run_fsdp(torch, dist, device, steps=args.steps, batch_size=args.batch_size)
                overlap_run = _overlap_probe(torch, dist, device)
                if args.config == "all":
                    for spec in _specs(world_size):
                        if spec.name == "fsdp":
                            continue
                        composition_runs[spec.name] = _run_composed(
                            torch, dist, device, spec, steps=args.steps, batch_size=args.batch_size)
                        if spec.pp_size > 1:
                            one_f_one_b = replace(spec, name=f"{spec.name}_1f1b", pp_schedule="1f1b")
                            composition_runs[one_f_one_b.name] = _run_composed(
                                torch, dist, device, one_f_one_b,
                                steps=args.steps, batch_size=args.batch_size)
            except Exception as exc:  # preserve the concrete server-side failure in the artifact
                fsdp_run = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                overlap_run = {"status": "failed", "reason": "overlap probe not reached after FSDP failure"}
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
        elif "fsdp" in selected:
            fsdp_run = {"status": "blocked", "reason": "FSDP correctness/performance requires torchrun world_size>=2"}
            overlap_run = {"status": "blocked", "reason": "overlap requires torchrun world_size>=2"}
    if rank == 0:
        payload = {"environment": {"world_size": world_size, "torch": getattr(locals().get("torch", None), "__version__", None),
                                    "cuda_available": bool(getattr(locals().get("torch", None), "cuda", None) and torch.cuda.is_available())},
                   "configuration": {"steps": args.steps, "batch_size": args.batch_size},
                   "matrix": matrix, "fsdp_run": fsdp_run, "composition_runs": composition_runs,
                   "overlap": overlap_run}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    failed_composition = any(item.get("status") == "failed" for item in composition_runs.values())
    return 0 if (fsdp_run is None or fsdp_run.get("status") in {"passed", "blocked"}) and not failed_composition else 1


if __name__ == "__main__":
    raise SystemExit(main())
