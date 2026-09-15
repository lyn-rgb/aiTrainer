"""Small, read-only CLI for validation, topology inspection and planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import FrameworkConfig
from .diagnostics import dry_run, inspect_topology, validate


def _load_config(path: str | None) -> FrameworkConfig:
    if path is None:
        return FrameworkConfig()
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read config {source}: {exc}") from exc
    return FrameworkConfig.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aitrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    topology = subparsers.add_parser("inspect-topology")
    topology.set_defaults(handler=lambda _: inspect_topology())
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--config")
    dry.add_argument("--world-size", type=int, default=None)
    dry.set_defaults(handler=lambda args: dry_run(_load_config(args.config), world_size=args.world_size))
    check = subparsers.add_parser("validate")
    check.add_argument("--config")
    check.add_argument("--world-size", type=int, default=1)
    def validate_handler(args: Any) -> dict[str, str]:
        validate(_load_config(args.config), world_size=args.world_size)
        return {"status": "ok"}
    check.set_defaults(handler=validate_handler)
    args = parser.parse_args(argv)
    try:
        output = args.handler(args)
    except Exception as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

