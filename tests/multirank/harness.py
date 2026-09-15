"""Launch a worker as N real OS processes and collect per-rank results.

``torchrun`` is unusable on this machine (its rendezvous resolves the local
hostname through mDNS and fails with ``gai error: 8``), but multi-rank itself is
not blocked: setting ``MASTER_ADDR``/``MASTER_PORT``/``RANK``/``WORLD_SIZE``
explicitly and calling ``init_process_group`` starts a working Gloo job.  The
distinction matters because it decides which fixes are actually verifiable.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().parent / "worker.py"


@dataclass(frozen=True)
class RankResult:
    rank: int
    returncode: int
    stdout: str
    stderr: str
    result: dict | None

    @property
    def error(self) -> str | None:
        for line in self.stdout.splitlines():
            if line.startswith("ERROR "):
                return line[len("ERROR "):]
        return None

    def describe(self) -> str:
        detail = self.error or f"exit {self.returncode}"
        return f"rank {self.rank}: {detail}\n--- stderr ---\n{self.stderr.strip()[-2000:]}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def run_case(case: str, world_size: int, *, shape: str | None = None,
             timeout_seconds: float = 30.0,
             hard_timeout: float = 120.0) -> list[RankResult]:
    """Run ``case`` on ``world_size`` processes and return every rank's result."""
    environment = dict(os.environ)
    environment.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_port()),
        "WORLD_SIZE": str(world_size),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
    })
    processes = []
    for rank in range(world_size):
        rank_environment = dict(environment, RANK=str(rank))
        command = [sys.executable, str(WORKER), "--case", case,
                   "--timeout-seconds", str(timeout_seconds)]
        if shape is not None:
            command += ["--shape", shape]
        processes.append(subprocess.Popen(
            command, cwd=str(REPO_ROOT), env=rank_environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    # One deadline for the whole job, not one per rank: a blocked rank is
    # exactly what several of these cases are looking for, and a per-process
    # timeout would multiply the wall clock by the world size.
    deadline = time.monotonic() + hard_timeout
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    results = []
    for rank, process in enumerate(processes):
        timed_out = process.poll() is None
        if timed_out:
            process.kill()
        stdout, stderr = process.communicate()
        if timed_out:
            stdout += f"\nERROR TimeoutExpired: no result within {hard_timeout}s"
        results.append(RankResult(rank, process.returncode, stdout, stderr, _parse(stdout)))
    return results


def require_success(results: list[RankResult]) -> list[dict]:
    """Assert every rank exited cleanly and return the parsed payloads."""
    failures = [r.describe() for r in results if r.returncode != 0 or r.result is None]
    assert not failures, "multi-rank case failed:\n" + "\n".join(failures)
    return [r.result for r in results]
