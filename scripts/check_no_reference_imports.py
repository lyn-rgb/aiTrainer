"""Fail if production code imports the bundled reference repositories."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).parents[1] / "src"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "references" or name.startswith("references.") for name in names):
                violations.append(f"{path}:{node.lineno}")
    if violations:
        print("reference imports found:", *violations, sep="\n", file=sys.stderr)
        return 1
    print("no reference repository imports in production code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
