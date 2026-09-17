#!/usr/bin/env bash
# Run examples/llama_shaped/train.py under every strategy, ONE AT A TIME.
#
# Run this in your own terminal, in the foreground.  It never backgrounds more
# than the ranks of a single strategy, and Ctrl-C stops it: `wait` returns on
# the interrupt and the trap below kills whatever ranks are still alive.  That
# is the whole reason this is a script you run rather than something the
# assistant runs for you -- a rank that blocks inside a collective has to be
# interruptible by the person watching it.
#
#   bash scripts/check_llama_shaped.sh              # every strategy
#   bash scripts/check_llama_shaped.sh tp pp        # only these
#   PYTHON=/path/to/venv/bin/python bash scripts/check_llama_shaped.sh
#
# The single-process strategy runs first: it is the reference the others are
# compared against, and it needs no process group at all.

set -u

cd "$(dirname "$0")/.." || exit 1
TRAIN="examples/llama_shaped/train.py"
# One port for the whole run; strategies are sequential and each one's ranks are
# reaped before the next starts.
PORT="${PORT:-29500}"

# strategy:world -- world is the product of the axes, declared not commanded.
ALL=(
  "single:1"
  "tp:2"
  "tp_sp:2"
  "pp:2"
  "fsdp:2"
  "dp2_tp2_sp:4"
)

cleanup() {
  echo
  echo "!! interrupted -- killing the ranks of the current strategy"
  pkill -f "$TRAIN" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

leftovers() { pgrep -f "$TRAIN" | wc -l | tr -d ' '; }

# Find an interpreter that actually has torch, rather than trusting `python3`.
# The first version of this script hardcoded `python3` and died immediately on a
# host where torch lives in a virtualenv -- which is every host this project is
# developed on, so the default was wrong for the only case that matters.
detect_python() {
  local candidates=()
  [ -n "${PYTHON:-}" ] && candidates+=("$PYTHON")
  [ -n "${PY:-}" ] && candidates+=("$PY")
  [ -x ".venv/bin/python" ] && candidates+=(".venv/bin/python")
  for name in python3 python; do
    command -v "$name" >/dev/null 2>&1 && candidates+=("$name")
  done
  for candidate in "${candidates[@]}"; do
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  echo "找不到带 torch 的解释器。试过:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  echo "用 PY=/path/to/venv/bin/python bash $0 指定。" >&2
  return 1
}

PYTHON="$(detect_python)" || exit 1

run_strategy() {
  local strategy="$1" world="$2"
  echo
  echo "==================================================================="
  echo "  $strategy   (world=$world, $(leftovers) stray ranks)"
  echo "==================================================================="
  if [ "$world" -eq 1 ]; then
    "$PYTHON" "$TRAIN" --strategy "$strategy"
    echo "  -> exit $?"
    return
  fi
  local pids=()
  for rank in $(seq 0 $((world - 1))); do
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
      WORLD_SIZE="$world" RANK="$rank" \
      "$PYTHON" "$TRAIN" --strategy "$strategy" --world "$world" &
    pids+=($!)
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  echo "  -> exit $status"
}

if [ "$#" -gt 0 ]; then
  selected=()
  for wanted in "$@"; do
    for entry in "${ALL[@]}"; do
      [ "${entry%%:*}" = "$wanted" ] && selected+=("$entry")
    done
  done
  [ "${#selected[@]}" -eq 0 ] && { echo "no such strategy; pick from: ${ALL[*]%%:*}"; exit 2; }
  ALL=("${selected[@]}")
fi

echo "python: $PYTHON  ($("$PYTHON" -c 'import torch; print("torch " + torch.__version__)'))"
if [ "$(leftovers)" != "0" ]; then
  echo "refusing to start: $(leftovers) process(es) already running $TRAIN"
  pgrep -fl "$TRAIN"
  exit 1
fi

for entry in "${ALL[@]}"; do
  run_strategy "${entry%%:*}" "${entry##*:}"
done

echo
echo "所有策略跑完。single 的 loss 是本清单的基准，其余应当与它一致。"
