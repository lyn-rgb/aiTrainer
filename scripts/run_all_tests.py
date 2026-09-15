#!/usr/bin/env python3
"""Run the complete aiTrainer validation matrix and build a portable report.

The runner has no third-party dependency.  It invokes project tools when they
are installed, records unavailable environments as ``blocked``/``skipped``
instead of claiming success, and writes JSON/CSV/HTML artifacts under one
directory.  The HTML report is self-contained and can be opened on a server
or copied to a workstation without network access.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>failed|passed|skipped|xfailed|xpassed|error|errors)")


@dataclass
class TestResult:
    name: str
    category: str
    status: str
    command: list[str]
    returncode: int | None
    duration_seconds: float
    stdout_log: str = ""
    stderr_log: str = ""
    reason: str = ""
    counts: dict[str, int] = field(default_factory=dict)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tool(name: str) -> str | None:
    return shutil.which(name)


def _module_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False, "version": None, "cuda_available": False,
                            "cuda_version": None, "device_count": 0, "devices": []}
    try:
        import torch
    except ImportError as exc:
        info["error"] = str(exc)
        return info
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": properties.name,
                            "total_memory_bytes": int(properties.total_memory),
                            "multi_processor_count": int(properties.multi_processor_count)})
    info.update({"installed": True, "version": getattr(torch, "__version__", None),
                 "cuda_available": bool(torch.cuda.is_available()),
                 "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                 "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                 "devices": devices})
    return info


def _parse_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _COUNT_RE.finditer(text):
        kind = match.group("kind")
        if kind == "errors":
            kind = "error"
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


class TestRunner:
    def __init__(self, report_dir: Path, *, strict: bool = False) -> None:
        self.report_dir = report_dir
        self.log_dir = report_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self.results: list[TestResult] = []
        self.benchmark_data: dict[str, Any] | None = None
        self.distributed_data: dict[str, Any] | None = None
        self.parallel_matrix_data: dict[str, Any] | None = None
        self.env = {"python": sys.version, "platform": platform.platform(),
                    "cwd": str(ROOT), "torch": _torch_info(),
                    "pytest": _module_available("pytest"),
                    "ruff": bool(_tool("ruff")), "mypy": bool(_tool("mypy")),
                    "torchrun": bool(_tool("torchrun"))}
        pycache = report_dir / "pycache"
        self.env_vars = os.environ.copy()
        self.env_vars["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + self.env_vars.get("PYTHONPATH", "")
        self.env_vars["PYTHONPYCACHEPREFIX"] = str(pycache)

    def _record(self, result: TestResult) -> None:
        self.results.append(result)
        state = result.status.upper()
        suffix = f" ({result.reason})" if result.reason else ""
        print(f"[{state:7}] {result.name} {result.duration_seconds:.2f}s{suffix}")

    def blocked(self, name: str, category: str, reason: str) -> None:
        self._record(TestResult(name, category, "blocked", [], None, 0.0, reason=reason))

    def skipped(self, name: str, category: str, reason: str) -> None:
        self._record(TestResult(name, category, "skipped", [], None, 0.0, reason=reason))

    def command(self, name: str, category: str, argv: Sequence[str], *, required: bool = True) -> None:
        command = [str(item) for item in argv]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=self.env_vars,
                                       text=True, capture_output=True, check=False)
            returncode = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except OSError as exc:
            duration = time.perf_counter() - started
            self._record(TestResult(name, category, "blocked" if required else "skipped", command,
                                    None, duration, reason=str(exc)))
            return
        duration = time.perf_counter() - started
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        stdout_path = self.log_dir / f"{stem}.stdout.log"
        stderr_path = self.log_dir / f"{stem}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        counts = _parse_counts(stdout + "\n" + stderr)
        status = "passed" if returncode == 0 else "failed"
        self._record(TestResult(name, category, status, command, returncode, duration,
                                str(stdout_path.relative_to(self.report_dir)),
                                str(stderr_path.relative_to(self.report_dir)), counts=counts))

    def write_json(self) -> Path:
        target = self.report_dir / "results.json"
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
                   "root": str(ROOT), "environment": self.env,
                   "results": [asdict(result) for result in self.results],
                   "summary": summarize(self.results), "training_benchmark": self.benchmark_data,
                   "distributed_correctness": self.distributed_data,
                   "parallel_matrix": self.parallel_matrix_data}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return target

    def write_csv(self) -> Path:
        target = self.report_dir / "results.csv"
        fields = ["name", "category", "status", "command", "returncode", "duration_seconds", "reason", "counts",
                  "stdout_log", "stderr_log"]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in self.results:
                row = asdict(result)
                row["command"] = " ".join(result.command)
                row["counts"] = json.dumps(result.counts, sort_keys=True)
                writer.writerow({field: row.get(field, "") for field in fields})
        return target

    def write_html(self) -> Path:
        target = self.report_dir / "report.html"
        summary = summarize(self.results)
        total = max(1, len(self.results))
        rows = []
        for result in self.results:
            command = html.escape(" ".join(result.command))
            logs = ""
            if result.stdout_log:
                logs += f'<a href="{html.escape(result.stdout_log)}">stdout</a> '
            if result.stderr_log:
                logs += f'<a href="{html.escape(result.stderr_log)}">stderr</a>'
            rows.append("<tr>" + "".join((
                f"<td>{html.escape(result.category)}</td>", f"<td>{html.escape(result.name)}</td>",
                f'<td><span class="{html.escape(result.status)}">{html.escape(result.status)}</span></td>',
                f"<td>{result.duration_seconds:.2f}</td>", f"<td>{result.returncode if result.returncode is not None else ''}</td>",
                f"<td><code>{command}</code><br>{html.escape(result.reason)}</td>",
                f"<td>{logs}</td>", "</tr>")))
        bars = []
        for status, color in (("passed", "#20a464"), ("failed", "#d64545"), ("blocked", "#d98c10"), ("skipped", "#7b8794")):
            count = summary.get(status, 0)
            width = 100.0 * count / total
            bars.append(f'<div class="bar" style="width:{width:.2f}%;background:{color}" title="{status}: {count}"></div>')
        generated = html.escape(datetime.now(timezone.utc).isoformat())
        benchmark_section = self._benchmark_html()
        distributed_section = self._distributed_html()
        matrix_section = self._parallel_matrix_html()
        target.write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>aiTrainer test report</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#202124}}h1{{margin-bottom:.25rem}}.meta{{color:#5f6368}}.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}.card{{padding:1rem 1.4rem;border-radius:8px;background:#f1f3f4;min-width:90px}}.value{{font-size:1.7rem;font-weight:700}}.passed{{color:#147d4f}}.failed{{color:#bd1e2d}}.blocked{{color:#9b6100}}.skipped{{color:#65727e}}.chart{{display:flex;height:26px;width:100%;background:#eee;border-radius:5px;overflow:hidden;margin:1rem 0}}.bar{{height:100%}}.metric{{height:24px;background:#edf0f2;border-radius:4px;overflow:hidden;margin:.25rem 0 1rem;max-width:760px}}.metric span{{display:block;height:100%;background:#3478c9;color:white;padding:3px 8px;box-sizing:border-box;white-space:nowrap}}.metric.memory span{{background:#7651a9}}table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}}th{{background:#f8f9fa}}code{{white-space:pre-wrap;word-break:break-word}}a{{margin-right:.4rem}}</style></head>
<body><h1>aiTrainer validation report</h1><div class="meta">Generated {generated}</div>
<div class="cards">{''.join(f'<div class="card"><div>{status}</div><div class="value {status}">{summary.get(status, 0)}</div></div>' for status in ("passed", "failed", "blocked", "skipped"))}<div class="card"><div>duration</div><div class="value">{summary.get("duration_seconds", 0.0):.1f}s</div></div></div>
<div class="chart">{"".join(bars)}</div><p>Exit recommendation: <strong>{"PASS" if summary.get("failed", 0) == 0 and summary.get("blocked", 0) == 0 else "ACTION REQUIRED"}</strong>. Blocked means the required environment or tool was unavailable.</p>
<h2>Environment</h2><pre>{html.escape(json.dumps(self.env, indent=2, default=str))}</pre>{distributed_section}{benchmark_section}{matrix_section}
<h2>Tests</h2><table><thead><tr><th>category</th><th>name</th><th>status</th><th>seconds</th><th>exit</th><th>command/reason</th><th>logs</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>\n''', encoding="utf-8")
        return target

    def _distributed_html(self) -> str:
        if not self.distributed_data:
            return "<h2>Distributed correctness</h2><p>No torchrun correctness result was produced.</p>"
        data = html.escape(json.dumps(self.distributed_data, indent=2, sort_keys=True))
        state = "passed" if self.distributed_data.get("passed") else "failed"
        return f'<h2>Distributed correctness</h2><p>Status: <strong class="{state}">{state}</strong></p><pre>{data}</pre>'

    def _benchmark_html(self) -> str:
        if not self.benchmark_data:
            return "<h2>Training benchmark</h2><p>No benchmark data was produced.</p>"
        runs = self.benchmark_data.get("runs", [])
        max_throughput = max((float(item.get("samples_per_second", 0.0)) for item in runs), default=1.0) or 1.0
        memory_values = [int(item.get("cuda_peak_allocated_bytes") or 0) for item in runs]
        max_memory = max(memory_values, default=1) or 1
        rows = []
        charts = []
        for item in runs:
            name = html.escape(str(item.get("name", "unknown")))
            throughput = float(item.get("samples_per_second", 0.0))
            peak = item.get("cuda_peak_allocated_bytes")
            peak_label = "n/a" if peak is None else f"{int(peak) / 1024 ** 2:.2f} MiB"
            rows.append(f"<tr><td>{name}</td><td>{float(item.get('steps_per_second', 0.0)):.3f}</td>"
                        f"<td>{throughput:.3f}</td><td>{html.escape(peak_label)}</td>"
                        f"<td>{int(item.get('cpu_current_rss_bytes', item.get('cpu_max_rss_bytes', 0))) / 1024 ** 2:.2f} MiB</td>"
                        f"<td>{float(item.get('final_loss', float('nan'))):.6f}</td></tr>")
            width = 100.0 * throughput / max_throughput
            memory_width = 0.0 if peak is None else 100.0 * int(peak) / max_memory
            charts.append(f'<div><strong>{name}</strong><div class="metric"><span style="width:{width:.2f}%">throughput {throughput:.2f} samples/s</span></div>'
                          f'<div class="metric memory"><span style="width:{memory_width:.2f}%">peak memory {html.escape(peak_label)}</span></div></div>')
        correctness = self.benchmark_data.get("correctness", {})
        return ("<h2>Training correctness, efficiency and memory</h2>"
                f"<p>Forward/backward correctness: <strong>{html.escape(str(correctness.get('passed', False)))}</strong>; "
                f"max output error={float(correctness.get('max_output_error', 0.0)):.3e}; "
                f"max gradient error={float(correctness.get('max_gradient_error', 0.0)):.3e}</p>"
                + "".join(charts) +
                "<table><thead><tr><th>variant</th><th>steps/s</th><th>samples/s</th><th>CUDA peak allocated</th><th>CPU current RSS</th><th>final loss</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>")

    def _parallel_matrix_html(self) -> str:
        if not self.parallel_matrix_data:
            return "<h2>Parallel combination matrix</h2><p>No FSDP/SP/TP/PP matrix result was produced.</p>"
        data = self.parallel_matrix_data
        rows = []
        for item in data.get("matrix", []):
            config = item.get("config", {}).get("parallel", {})
            rows.append("<tr>" + "".join((
                f"<td>{html.escape(str(item.get('name', '')))}</td>",
                f"<td>{html.escape(str(item.get('status', '')))}</td>",
                f"<td>{html.escape(str(item.get('expected_status', '')))}</td>",
                f"<td>dp={config.get('dp_size')} tp={config.get('tp_size')} pp={config.get('pp_size')} sp={html.escape(str(config.get('sp_backend')))}</td>",
                f"<td>{html.escape(str(item.get('reason', '')))}</td>", "</tr>")))
        run = data.get("fsdp_run") or {}
        compositions = data.get("composition_runs") or {}
        overlap = data.get("overlap") or {}
        memory = run.get("max_peak_allocated_bytes")
        memory_label = "n/a" if memory is None else f"{int(memory) / 1024 ** 2:.2f} MiB"
        memory_rows = []
        for rank_memory in run.get("per_rank_memory", []):
            peak_rank = rank_memory.get("peak_allocated_bytes")
            reserved_rank = rank_memory.get("peak_reserved_bytes")
            peak_text = "n/a" if peak_rank is None else f"{int(peak_rank) / 1024 ** 2:.2f} MiB"
            reserved_text = "n/a" if reserved_rank is None else f"{int(reserved_rank) / 1024 ** 2:.2f} MiB"
            memory_rows.append(f"<tr><td>{html.escape(str(rank_memory.get('rank', '')))}</td><td>{peak_text}</td><td>{reserved_text}</td></tr>")
        state = html.escape(str(run.get("status", "blocked")))
        composition_rows = []
        for name, value in compositions.items():
            composition_peak = value.get("max_peak_allocated_bytes")
            composition_peak_label = "n/a" if composition_peak is None else f"{int(composition_peak) / 1024 ** 2:.2f} MiB"
            composition_rows.append(
                f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(value.get('status', '')))}</td>"
                f"<td>{float(value.get('loss', 0.0)):.6f}</td><td>{float(value.get('global_samples_per_second', 0.0)):.3f}</td>"
                f"<td>{html.escape(composition_peak_label)}</td>"
                f"<td>{html.escape(str(value.get('reason', '')))}</td></tr>")
        return ("<h2>FSDP/SP/TP/PP combination matrix</h2>"
                "<p>Each requested composition is validated before runtime; failed or unavailable entries retain their concrete reason.</p>"
                "<table><thead><tr><th>configuration</th><th>actual</th><th>expected</th><th>topology</th><th>reason</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>"
                + f"<h3>FSDP runtime</h3><p>Status: <strong class=\"{state}\">{state}</strong>; "
                f"correctness output error={float(run.get('output_max_abs_error', 0.0)):.3e}, loss error={float(run.get('loss_abs_error', 0.0)):.3e}, parameter error={float(run.get('parameter_max_abs_error', 0.0)):.3e}; "
                f"global throughput={float(run.get('global_samples_per_second', 0.0)):.3f} samples/s; max per-card peak allocated={html.escape(memory_label)}</p>"
                "<h4>Per-rank CUDA memory</h4><table><thead><tr><th>rank</th><th>peak allocated</th><th>peak reserved</th></tr></thead><tbody>"
                + ("".join(memory_rows) or "<tr><td colspan=3>n/a</td></tr>") + "</tbody></table>"
                "<h4>Combined composition smoke runs</h4><table><thead><tr><th>configuration</th><th>status</th><th>last loss</th><th>samples/s</th><th>peak allocated</th><th>reason</th></tr></thead><tbody>"
                + ("".join(composition_rows) or "<tr><td colspan=6>n/a</td></tr>") + "</tbody></table>"
                f"<h3>Calculation/communication overlap probe</h3><pre>{html.escape(json.dumps(overlap, indent=2, sort_keys=True, default=str))}</pre>")


def summarize(results: Sequence[TestResult]) -> dict[str, int | float]:
    summary: dict[str, int | float] = {"total": len(results), "passed": 0, "failed": 0,
                                       "blocked": 0, "skipped": 0, "duration_seconds": 0.0}
    for result in results:
        summary[result.status] = int(summary.get(result.status, 0)) + 1
        summary["duration_seconds"] = float(summary["duration_seconds"]) + result.duration_seconds
    return summary


def _pytest_command(args: argparse.Namespace, paths: Sequence[str]) -> list[str] | None:
    if not _module_available("pytest"):
        return None
    command = [sys.executable, "-m", "pytest", "-q", *paths]
    if args.pytest_args:
        command.extend(args.pytest_args)
    return command


def run(args: argparse.Namespace) -> int:
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else ROOT / "artifacts" / f"test_report_{_timestamp()}"
    runner = TestRunner(report_dir, strict=args.strict)
    print(f"aiTrainer root: {ROOT}\nreport directory: {report_dir}\n")
    runner.command("compileall", "static", [sys.executable, "-m", "compileall", "-q", "src", "examples", "tests"])
    runner.command("reference_import_check", "static", [sys.executable, "scripts/check_no_reference_imports.py"])
    runner.command("package_import_smoke", "static", [sys.executable, "-c", "import aitrainer; print('import ok')"])
    if args.skip_overlap_smoke:
        runner.skipped("overlap_contract_smoke", "overlap", "disabled by --skip-overlap-smoke")
    else:
        overlap_output = report_dir / "overlap_smoke.json"
        runner.command("overlap_contract_smoke", "overlap",
                       [sys.executable, "scripts/overlap_smoke.py", "--output", str(overlap_output)])
    if _tool("ruff"):
        runner.command("ruff", "static", ["ruff", "check", "src", "tests", "examples"], required=False)
    else:
        runner.skipped("ruff", "static", "ruff is not installed")
    if _tool("mypy"):
        runner.command("mypy", "static", ["mypy", "src/aitrainer"], required=False)
    else:
        runner.skipped("mypy", "static", "mypy is not installed")
    pytest_command = _pytest_command(args, [])
    if pytest_command is None:
        runner.blocked("pytest_unit", "unit", "pytest runner is unavailable")
    else:
        unit = _pytest_command(args, ["tests/unit"])
        runner.command("pytest_unit", "unit", unit or [], required=True)
        boundary = _pytest_command(args, ["tests/distributed"])
        runner.command("pytest_distributed_boundary", "distributed-boundary", boundary or [], required=True)
        if not args.skip_performance:
            performance = _pytest_command(args, ["tests/performance"])
            runner.command("pytest_performance_baseline", "performance", performance or [], required=True)
        else:
            runner.skipped("pytest_performance_baseline", "performance", "disabled by --skip-performance")
    torch = runner.env["torch"]
    if args.skip_benchmark:
        runner.skipped("training_efficiency_memory", "training-benchmark", "disabled by --skip-benchmark")
    elif not torch["installed"]:
        runner.blocked("training_efficiency_memory", "training-benchmark", "PyTorch is not installed")
    else:
        benchmark_output = report_dir / "training_benchmark.json"
        runner.command("training_efficiency_memory", "training-benchmark",
                       [sys.executable, "scripts/benchmark_training.py", "--output", str(benchmark_output),
                        "--steps", str(args.benchmark_steps), "--batch-size", str(args.benchmark_batch_size),
                        "--repeats", str(args.benchmark_repeats)])
        if benchmark_output.is_file():
            try:
                runner.benchmark_data = json.loads(benchmark_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                runner.benchmark_data = None
        if not torch["cuda_available"]:
            runner.blocked("cuda_memory_measurement", "training-benchmark",
                           "CUDA is unavailable; CPU benchmark ran but VRAM measurement was not executed")
    if args.skip_distributed:
        runner.skipped("torchrun_distributed", "distributed", "disabled by --skip-distributed")
    elif not torch["installed"]:
        runner.blocked("torchrun_distributed", "distributed", "PyTorch is not installed")
    elif not runner.env["pytest"]:
        runner.blocked("torchrun_distributed", "distributed", "pytest is not installed")
    else:
        torchrun = _tool("torchrun")
        if torchrun is None:
            launcher = [sys.executable, "-m", "torch.distributed.run"]
        else:
            launcher = [torchrun]
        launch_args = ["--standalone", "--nproc_per_node", str(args.world_size)]
        distributed = launcher + launch_args + ["-m", "pytest", "-q", "tests/distributed"]
        runner.command(f"torchrun_distributed_{args.world_size}r", "distributed", distributed, required=True)
        distributed_correctness = launcher + launch_args + [str(ROOT / "scripts/distributed_correctness.py")]
        runner.command(f"torchrun_forward_backward_{args.world_size}r", "distributed-correctness",
                       distributed_correctness, required=True)
        if runner.results and runner.results[-1].stdout_log:
            log_path = runner.report_dir / runner.results[-1].stdout_log
            try:
                for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
                    if line.strip().startswith("{"):
                        runner.distributed_data = json.loads(line)
                        break
            except (OSError, json.JSONDecodeError):
                runner.distributed_data = None
    if args.skip_parallel_matrix:
        runner.skipped("parallel_combination_matrix", "parallel-matrix", "disabled by --skip-parallel-matrix")
    elif not torch["installed"]:
        runner.blocked("parallel_combination_matrix", "parallel-matrix", "PyTorch is not installed")
    else:
        torchrun = _tool("torchrun")
        matrix_launcher = [torchrun] if torchrun else [sys.executable, "-m", "torch.distributed.run"]
        matrix_output = report_dir / "parallel_matrix.json"
        matrix_command = (matrix_launcher + ["--standalone", "--nproc_per_node", str(args.world_size),
                         str(ROOT / "scripts/parallel_matrix.py"), "--output", str(matrix_output),
                         "--steps", str(args.parallel_matrix_steps), "--batch-size", str(args.benchmark_batch_size)])
        runner.command(f"parallel_combination_matrix_{args.world_size}r", "parallel-matrix", matrix_command, required=True)
        try:
            if matrix_output.is_file():
                runner.parallel_matrix_data = json.loads(matrix_output.read_text(encoding="utf-8"))
            else:
                matrix_result = runner.results[-1]
                if matrix_result.stdout_log:
                    log_path = runner.report_dir / matrix_result.stdout_log
                    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
                        if line.strip().startswith("{"):
                            runner.parallel_matrix_data = json.loads(line)
                            break
        except (OSError, json.JSONDecodeError):
            runner.parallel_matrix_data = None
    runner.write_json()
    runner.write_csv()
    report = runner.write_html()
    summary = summarize(runner.results)
    print(f"\nHTML report: {report}\nJSON: {report.parent / 'results.json'}\nCSV: {report.parent / 'results.csv'}")
    print("Summary:", json.dumps(summary, sort_keys=True))
    return 1 if summary["failed"] or summary["blocked"] or (args.strict and summary["skipped"]) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", help="output directory (default: artifacts/test_report_<UTC timestamp>)")
    parser.add_argument("--world-size", type=int, default=2, help="torchrun process count (default: 2)")
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--benchmark-batch-size", type=int, default=32)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--parallel-matrix-steps", type=int, default=10,
                        help="timed FSDP steps in the parallel combination matrix")
    parser.add_argument("--skip-parallel-matrix", action="store_true")
    parser.add_argument("--skip-overlap-smoke", action="store_true",
                        help="skip dependency-free Batch 9 overlap lifecycle smoke")
    parser.add_argument("--strict", action="store_true", help="return nonzero for skipped/blocked checks too")
    parser.add_argument("--pytest-args", nargs="*", default=[], help="extra arguments appended to pytest commands")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.world_size < 2:
        parser.error("--world-size must be >= 2 for the distributed phase")
    if args.parallel_matrix_steps < 1:
        parser.error("--parallel-matrix-steps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
