"""``import aitrainer`` must work with no torch installed.

This property was always true but always incidental: it held because every
module happened to guard its imports, in one of nine different shapes, and
nothing ever checked it.  A single unguarded module-level ``import torch`` added
anywhere would have broken it silently on any machine that does not train --
which is most machines that merely *validate* a config or inspect a topology.

The test blocks ``torch`` at the import machinery rather than uninstalling it, so
it runs everywhere and also covers the ``find_spec`` path used by
``core.torch.torch_available``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK_TORCH = """
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _BlockTorch())
for module in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
    del sys.modules[module]

import aitrainer

missing = [name for name in aitrainer.__all__ if not hasattr(aitrainer, name)]
assert not missing, f"names in __all__ but not importable: {missing}"
print("OK", len(aitrainer.__all__))
"""


def test_package_imports_and_exports_resolve_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        "importing aitrainer with torch blocked failed:\n"
        f"{result.stdout}\n{result.stderr}")
    assert result.stdout.startswith("OK "), result.stdout


@pytest.mark.parametrize("helper", ["torch_available", "dist", "is_distributed", "world_size", "rank"])
def test_core_torch_helpers_degrade_without_torch(helper):
    """The single optional-torch policy must answer without importing torch."""
    program = _BLOCK_TORCH + (
        f"\nfrom aitrainer.core import torch as core_torch\n"
        f"print('VALUE', core_torch.{helper}())\n")
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "VALUE" in result.stdout, result.stdout
