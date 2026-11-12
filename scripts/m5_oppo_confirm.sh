#!/usr/bin/env bash
# OPPO-MC confirmation (hns.marl.oppo_block; declared in its docstring), on the JarvisLabs A30:
#   1. seeds 1 and 2 against both frozen opponents, and the scripted references on the confirmation worlds,
#      six processes at once under CUDA MPS;
#   2. every run's evaluation on worlds 200-207 (seeds 0, 1, 2 x both opponents), six at once;
#   3. the reading: runs/repair2/oppo_confirmation.json.
# Seed 0's development runs (runs/repair2/OPPO_{full0,nda1}_s0) and the opponents (runs/marl/{full/0,nda/1}) must be
# on the instance. Every process has a 2-hour timeout. Every step runs even if an earlier one failed (the reading
# marks what is missing as incomplete); the script exits non-zero if any step failed.
#   scripts/m5_oppo_confirm.sh [smoke]      (smoke: 2 training worlds, every seed, 1 evaluation world; runs/repair2/smoke)
#   logs: runs/repair2/logs/confirm_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/remote_env.sh ] && . scripts/remote_env.sh
LOGS=runs/repair2/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-cuda-mps-control -d 2>/dev/null || true          # several processes on one GPU (2026-11-10: about 2x)
if [ "${1:-}" = smoke ]; then
  P=smoke_; SEEDS_TRAIN="0 1 2"; RUN=(--worlds 2 --eval-worlds 1); TRAIN=(--worlds 2); REF=(--eval-worlds 1)
else
  P=; SEEDS_TRAIN="1 2"; RUN=(); TRAIN=(); REF=()
fi
FAILED=0 PIDS=() NAMES=()
job() {  # name, module and arguments: start it in the background
  local name=$1; shift
  timeout 7200 uv run python -m "$@" >> "$LOGS/confirm_$P$name.log" 2>&1 &
  PIDS+=($!) NAMES+=("$name")
}
barrier() {  # wait for every started job, one by one, and count the failures
  local i rc
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; rc=$?
    if [ "$rc" -ne 0 ]; then echo "FAILED (exit $rc$([ "$rc" -eq 124 ] && echo ', timeout')): ${NAMES[$i]}"; FAILED=$((FAILED + 1)); fi
  done
  PIDS=() NAMES=()
}
for o in full0 nda1; do
  for s in $SEEDS_TRAIN; do job "train_${o}_s$s" hns.marl.oppo_block train --opp "$o" --seed "$s" "${TRAIN[@]}"; done
  job "refs_$o" hns.marl.repair2 refs --opp "$o" --worlds confirmation "${REF[@]}"
done
barrier
for o in full0 nda1; do
  for s in 0 1 2; do job "eval_${o}_s$s" hns.marl.oppo_block confirm-eval --opp "$o" --seed "$s" "${RUN[@]}"; done
done
barrier
uv run python -m hns.marl.oppo_block confirm "${RUN[@]}" > "$LOGS/confirm_${P}reading.log" 2>&1 \
  || { echo "FAILED: reading"; FAILED=$((FAILED + 1)); }
echo "confirmation finished: $(date), $FAILED failed step(s)"
[ "$FAILED" -eq 0 ]
