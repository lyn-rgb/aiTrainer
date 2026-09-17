#!/usr/bin/env bash
# Create the project virtualenv at <repo>/.venv and install everything.
#
#   bash scripts/create_venv.sh                 # torch from PyPI (CUDA build on Linux)
#   bash scripts/create_venv.sh --cuda cu126    # a specific CUDA wheel line
#   bash scripts/create_venv.sh --cpu           # CPU-only wheels (smaller download)
#   bash scripts/create_venv.sh --force         # replace an existing .venv
#   bash scripts/create_venv.sh --no-smoke      # skip the 3-step training check
#
# The project has exactly ONE runtime dependency: torch.  Everything else in
# pyproject.toml's `dev` extra -- pytest, ruff, mypy, import-linter -- is for
# the test suite and the static checks, not for training.
#
# What the GPU server has to provide is a DRIVER, not a toolkit.  PyTorch's
# Linux wheels bundle the CUDA runtime and NCCL, so `nvidia-smi` reporting a
# new-enough driver is the whole requirement; installing CUDA separately is
# only needed if you build something else against it.  The wheel line has to
# match or predate the driver: a cu126 wheel on a driver that only supports
# CUDA 12.4 fails at import, not at install.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
VENV="$REPO/.venv"

CUDA=""
CPU_ONLY=0
FORCE=0
SMOKE=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda) CUDA="${2:?--cuda needs a value, e.g. cu126}"; shift 2 ;;
    --cpu) CPU_ONLY=1; shift ;;
    --force) FORCE=1; shift ;;
    --no-smoke) SMOKE=0; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$CPU_ONLY" -eq 1 ] && [ -n "$CUDA" ]; then
  echo "--cpu and --cuda are mutually exclusive" >&2
  exit 2
fi

# --- an interpreter that satisfies requires-python ------------------------ #
# pyproject says >=3.10.  Prefer the newest, because torch ships wheels for the
# current few and a too-new interpreter is the usual reason an install finds
# nothing.
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "找不到 Python >= 3.10。已试过: python3.13 python3.12 python3.11 python3.10 python3" >&2
  exit 1
fi
echo "使用解释器: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

# --- the venv ------------------------------------------------------------- #
if [ -e "$VENV" ]; then
  if [ "$FORCE" -eq 1 ]; then
    echo "移除已存在的 $VENV"
    rm -rf "$VENV"
  else
    echo "$VENV 已存在。要重建请加 --force（会先删除整个目录）。" >&2
    exit 1
  fi
fi

echo "创建 $VENV"
"$PYTHON" -m venv --upgrade-deps "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --quiet --upgrade pip setuptools wheel

# --- torch ---------------------------------------------------------------- #
# Installed on its own, from the torch index when one is given: passing
# --index-url to the project install below would send pytest and ruff to an
# index that does not carry them.
if [ "$CPU_ONLY" -eq 1 ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
elif [ -n "$CUDA" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"
else
  TORCH_INDEX=""
fi

if [ -n "$TORCH_INDEX" ]; then
  echo "安装 torch（索引 $TORCH_INDEX）—— 这是最大的一步，几分钟"
  "$PIP" install --index-url "$TORCH_INDEX" "torch>=2.1"
else
  echo "安装 torch（PyPI 默认轮子）—— 这是最大的一步，几分钟"
  "$PIP" install "torch>=2.1"
fi

# --- the project and its dev extra ---------------------------------------- #
echo "安装 aitrainer 及其 dev 依赖"
"$PIP" install --editable "$REPO[dev]"

# --- report what actually landed ------------------------------------------ #
echo
echo "==================================================================="
"$VENV/bin/python" - <<'PY'
import sys
import torch

print(f"python          {sys.version.split()[0]}")
print(f"torch           {torch.__version__}")
print(f"torch 构建 CUDA  {torch.version.cuda or '(无，CPU 版)'}")
# Guarded: a CPU build has no torch.cuda.nccl to ask, and NCCL is what every
# multi-rank path in this project needs (all_to_all has no Gloo implementation,
# so the whole Ulysses family is CUDA-only).
try:
    nccl = torch.cuda.nccl.version()
except (AttributeError, RuntimeError):
    nccl = None
print(f"nccl 版本        {nccl or '(不可用 —— 多卡路径将不可用)'}")
print(f"CUDA 可用        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"设备数           {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"  [{index}] {torch.cuda.get_device_name(index)}")
else:
    print("设备数           0 —— 多卡相关的一切都无法验证，见 docs/GPU验证清单.md")
PY
echo "==================================================================="

# --- prove it trains ------------------------------------------------------- #
if [ "$SMOKE" -eq 1 ]; then
  echo
  echo "冒烟检查：单进程训练 3 步"
  "$VENV/bin/python" examples/llama_shaped/train.py --strategy single
fi

cat <<EOF

完成。启用：

    source .venv/bin/activate

多 rank 由你自己起进程（本仓库的脚本都不自己起进程）：

    for r in 0 1; do MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=\$r \\
        .venv/bin/python examples/llama_shaped/train.py --strategy fsdp --world 2 & done; wait

CPU 版 wheel 的机器上只能跑单卡；其余能力清单见 docs/使用指南.md §6 与 docs/GPU验证清单.md。
EOF
