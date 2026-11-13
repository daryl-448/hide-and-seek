#!/usr/bin/env bash
# OPPO-MC-60R (hns.marl.oppo_block --attempt 60r; declared in its docstring), on the JarvisLabs A30:
#   1. seeds 6, 7 and 8 against both opponents (60 worlds each), and the scripted references on worlds 224-239,
#      at once under CUDA MPS (two-hour timeout each); there is no development block;
#   2. every run's evaluation on worlds 224-239 and its state-use readout, one fresh process at a time;
#   3. the reading: runs/repair2/oppo60r_confirmation.json.
# Needs on the instance: the opponents (runs/marl/{full/0,nda/1}: config.yaml, ckpt/final.pt).
# Every step runs even if an earlier one failed (the reading marks what is missing as incomplete); the script exits
# non-zero if any step failed.
#   scripts/m5_oppo60r.sh [smoke]   (smoke: 2 training worlds, 1 evaluation world; runs/repair2/smoke)
#   logs: runs/repair2/logs/oppo60r_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/remote_env.sh ] && . scripts/remote_env.sh
LOGS=runs/repair2/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=(--attempt 60r)
if [ "${1:-}" = smoke ]; then
  P=smoke_; RUN=(--worlds 2 --eval-worlds 1); TRAIN=(--worlds 2); REF=(--eval-worlds 1)
else
  P=; RUN=(); TRAIN=(); REF=()
fi
FAILED=0 PIDS=() NAMES=()
fail() { echo "FAILED (exit $1$([ "$1" -eq 124 ] && echo ', timeout')): $2"; FAILED=$((FAILED + 1)); }
step() {  # name, timeout in seconds, module and arguments: run it in the foreground
  local name=$1 t=$2 rc; shift 2
  timeout "$t" uv run python -m "$@" >> "$LOGS/oppo60r_$P$name.log" 2>&1; rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "$name"
}
job() {  # name, module and arguments: start it in the background
  local name=$1; shift
  timeout 7200 uv run python -m "$@" >> "$LOGS/oppo60r_$P$name.log" 2>&1 &
  PIDS+=($!) NAMES+=("$name")
}
barrier() {  # wait for every started job, one by one, and count the failures
  local i rc
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; rc=$?
    [ "$rc" -eq 0 ] || fail "$rc" "${NAMES[$i]}"
  done
  PIDS=() NAMES=()
}
nvidia-cuda-mps-control -d 2>/dev/null || true          # several processes on one GPU (2026-11-10: about 2x)
for o in full0 nda1; do
  for s in 6 7 8; do job "train_${o}_s$s" hns.marl.oppo_block train "${A[@]}" --opp "$o" --seed "$s" "${TRAIN[@]}"; done
  job "refs_$o" hns.marl.repair2 refs --opp "$o" --worlds confirmation60r "${REF[@]}"
done
barrier
for o in full0 nda1; do
  for s in 6 7 8; do step "eval_${o}_s$s" 3600 hns.marl.oppo_block confirm-eval "${A[@]}" --opp "$o" --seed "$s" "${RUN[@]}"; done
done
step reading 600 hns.marl.oppo_block confirm "${A[@]}" "${RUN[@]}"
# a caught abort exits 0 but leaves its opponent's reading incomplete: count that as a failed step
uv run python - "runs/repair2/${P:+smoke/}oppo60r_confirmation.json" <<'PY' || fail 1 "an incomplete reading"
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
st = {o: r["reading"]["status"] for o, r in json.loads(f.read_text())["opponents"].items()} if f.exists() else {}
print("\n".join(f"reading {o}: {s}" for o, s in st.items()) or f"no {f}")
sys.exit(0 if st and all(s == "ok" for s in st.values()) else 1)
PY
echo "OPPO-MC-60R finished: $(date), $FAILED failed step(s)"
[ "$FAILED" -eq 0 ]
