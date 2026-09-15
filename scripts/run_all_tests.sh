#!/usr/bin/env bash
# One-command validation entry point. Batch 9 overlap lifecycle smoke is
# enabled by default; CUDA/NCCL overlap and FSDP/SP/TP/PP matrix runs are
# launched automatically when PyTorch and torchrun are available.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# Honour $PYTHON so this runs under the interpreter that actually has PyTorch.
# A bare `python3` is frequently a system build without torch (on macOS it is
# 3.9), which would report the whole suite as blocked.
exec "${PYTHON:-python3}" "$SCRIPT_DIR/run_all_tests.py" "$@"
