#!/usr/bin/env python3
"""Two-or-more-rank forward/backward correctness check for torchrun."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError as exc:
        print(f"PyTorch distributed is required: {exc}", file=sys.stderr)
        return 2
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size < 2:
        print("WORLD_SIZE must be >= 2", file=sys.stderr)
        return 2
    use_cuda = bool(torch.cuda.is_available())
    backend = "nccl" if use_cuda else "gloo"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if use_cuda else torch.device("cpu")
    try:
        dist.init_process_group(backend=backend)
        torch.manual_seed(1701)
        reference = torch.nn.Linear(8, 4).to(device)
        candidate = torch.nn.Linear(8, 4).to(device)
        candidate.load_state_dict(reference.state_dict())
        wrapped = DDP(candidate, device_ids=[device.index] if use_cuda else None)
        inputs = torch.randn(6, 8, device=device)
        labels = torch.randn(6, 4, device=device)
        expected_output = reference(inputs)
        expected_loss = torch.nn.functional.mse_loss(expected_output, labels)
        expected_loss.backward()
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        reference_optimizer.step()
        actual_output = wrapped(inputs)
        actual_loss = torch.nn.functional.mse_loss(actual_output, labels)
        actual_loss.backward()
        output_error = float((actual_output.detach() - expected_output.detach()).abs().max().item())
        gradient_error = max(float((actual.grad - expected.grad).abs().max().item())
                             for actual, expected in zip(wrapped.module.parameters(), reference.parameters()))
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        optimizer.step()
        parameter_error_tensor = torch.tensor(
            [float((actual.detach() - expected.detach()).abs().max().item())
             for actual, expected in zip(wrapped.module.parameters(), reference.parameters())], device=device)
        dist.all_reduce(parameter_error_tensor, op=dist.ReduceOp.MAX)
        parameter_error = float(parameter_error_tensor.max().item())
        finite = bool(torch.isfinite(actual_loss).item()) and all(
            bool(torch.isfinite(parameter.grad).all().item()) for parameter in wrapped.module.parameters())
        passed = output_error <= 1e-6 and gradient_error <= 1e-6 and parameter_error <= 1e-6 and finite
        dist.barrier()
        if rank == 0:
            print(json.dumps({"passed": passed, "backend": backend, "world_size": world_size,
                              "output_max_abs_error": output_error,
                              "gradient_max_abs_error": gradient_error,
                              "parameter_max_abs_error": parameter_error,
                              "finite": finite}, sort_keys=True))
        return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
