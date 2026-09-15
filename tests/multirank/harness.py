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
    """A port that was free a moment ago, verified across several attempts.

    Binding to port 0 and closing hands the port straight back to the kernel's
    ephemeral pool, so a following attempt can receive it again.  Each candidate
    is re-bound to check it, which is not a proof -- nothing is, short of holding
    the socket -- but it makes an accidental collision rare.  A collision used to
    surface as a red gate that passed on re-run.
    """
    for _ in range(20):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port < 20000:          # avoid the low range other tools squat on
            continue
        with socket.socket() as verify:
            verify.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                verify.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("could not find a free port for the rendezvous")


# A rendezvous that never completed is an infrastructure failure, not a result.
# It is textually distinct from what the negative controls look for: a store
# *barrier* timeout after a successful rendezvous reports
# "wait timeout after Nms, keys: ...", while these report a connection that was
# never established.  Retrying only the former keeps the deadlock controls honest.
_RENDEZVOUS_FAILURES = (
    "waiting for clients",
    "client socket has timed out",
    "Address already in use",
    "Address in use",
)


def _rendezvous_failed(results: list["RankResult"]) -> bool:
    combined = " ".join((result.error or "") + result.stderr for result in results)
    return any(marker in combined for marker in _RENDEZVOUS_FAILURES)


def _parse(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def run_case(case: str, world_size: int, *, shape: str | None = None,
             options: dict[str, str] | None = None,
             timeout_seconds: float = 30.0,
             hard_timeout: float = 120.0) -> list[RankResult]:
    """Run ``case`` on ``world_size`` processes and return every rank's result.

    A rendezvous that never came up is retried once, because a transient port
    collision would otherwise surface as a red gate that passes on re-run.
    Cases that *expect* a timeout are unaffected -- see ``_RENDEZVOUS_FAILURES``.
    """
    results = _run_once(case, world_size, shape=shape, options=options,
                        timeout_seconds=timeout_seconds, hard_timeout=hard_timeout)
    if _rendezvous_failed(results):
        results = _run_once(case, world_size, shape=shape, options=options,
                            timeout_seconds=timeout_seconds, hard_timeout=hard_timeout)
    return results


def _run_once(case: str, world_size: int, *, shape: str | None,
              options: dict[str, str] | None, timeout_seconds: float,
              hard_timeout: float) -> list[RankResult]:
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
        for key, value in sorted((options or {}).items()):
            command += ["--option", f"{key}={value}"]
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
