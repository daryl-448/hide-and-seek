#!/usr/bin/env bash
# OPPO-MC-60 (hns.marl.oppo_block --attempt 60; declared in its docstring), on the JarvisLabs A30:
#   1. the development blocks, seed 3, Full-0 then NDA-1, one at a time, each one attempt under a 60-minute cap;
#   2. only if both read promising: seeds 4 and 5 against both opponents, and the scripted references on worlds
#      208-215, at once under CUDA MPS (two-hour timeout each);
#   3. every run's evaluation on worlds 208-215 (seeds 3, 4, 5 x both opponents), one fresh process at a time;
#   4. the reading: runs/repair2/oppo60_confirmation.json.
# Needs on the instance: the opponents (runs/marl/{full/0,nda/1}: config.yaml, ckpt/final.pt), the development
# references (runs/repair2/ref_*_selection_*.json) and legacy A's and PL's records (runs/repair2/{A,PL}_*_s0).
# After step 1 every step runs even if an earlier one failed (the reading marks what is missing as incomplete);
# the script exits non-zero if any step failed, and 3 if the development blocks did not both read promising.
#   scripts/m5_oppo60.sh [smoke]   (smoke: 2 training worlds, 1 evaluation world, no gate; runs/repair2/smoke)
#   logs: runs/repair2/logs/oppo60_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/remote_env.sh ] && . scripts/remote_env.sh
LOGS=runs/repair2/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=(--attempt 60)
if [ "${1:-}" = smoke ]; then
  P=smoke_; RUN=(--worlds 2 --eval-worlds 1); TRAIN=(--worlds 2); REF=(--eval-worlds 1)
else
  P=; RUN=(); TRAIN=(); REF=()
fi
FAILED=0 PIDS=() NAMES=()
fail() { echo "FAILED (exit $1$([ "$1" -eq 124 ] && echo ', timeout')): $2"; FAILED=$((FAILED + 1)); }
step() {  # name, timeout in seconds, module and arguments: run it in the foreground
  local name=$1 t=$2 rc; shift 2
  timeout "$t" uv run python -m "$@" >> "$LOGS/oppo60_$P$name.log" 2>&1; rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "$name"
}
job() {  # name, module and arguments: start it in the background
  local name=$1; shift
  timeout 7200 uv run python -m "$@" >> "$LOGS/oppo60_$P$name.log" 2>&1 &
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
for o in full0 nda1; do step "dev_$o" 3600 hns.marl.oppo_block run "${A[@]}" --opp "$o" "${RUN[@]}"; done
if [ -z "$P" ]; then                   # the confirmation runs only if both development blocks read promising
  if ! uv run python - <<'EOF'
import json, sys
from pathlib import Path
ok = []
for o in ("full0", "nda1"):
    f = Path(f"runs/repair2/oppo60_{o}_s3.json")
    m = json.loads(f.read_text()) if f.exists() else {}
    print(f"development {o}: {m.get('state', 'no marker')}, promising {m.get('promising')}")
    ok.append(m.get("state") == "done" and m.get("promising") is True)
sys.exit(0 if all(ok) else 1)
EOF
  then
    echo "OPPO-MC-60: the development blocks did not both read promising; the confirmation is not run ($(date))"
    [ "$FAILED" -eq 0 ] && exit 3 || exit 1
  fi
fi
nvidia-cuda-mps-control -d 2>/dev/null || true          # several processes on one GPU (2026-11-10: about 2x)
for o in full0 nda1; do
  for s in 4 5; do job "train_${o}_s$s" hns.marl.oppo_block train "${A[@]}" --opp "$o" --seed "$s" "${TRAIN[@]}"; done
  job "refs_$o" hns.marl.repair2 refs --opp "$o" --worlds confirmation60 "${REF[@]}"
done
barrier
for o in full0 nda1; do
  for s in 3 4 5; do step "eval_${o}_s$s" 2700 hns.marl.oppo_block confirm-eval "${A[@]}" --opp "$o" --seed "$s" "${RUN[@]}"; done
done
step reading 600 hns.marl.oppo_block confirm "${A[@]}" "${RUN[@]}"
# a caught abort exits 0 but leaves its opponent's reading incomplete: count that as a failed step
uv run python - "runs/repair2/${P:+smoke/}oppo60_confirmation.json" <<'EOF' || fail 1 "an incomplete reading"
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
st = {o: r["reading"]["status"] for o, r in json.loads(f.read_text())["opponents"].items()} if f.exists() else {}
print("\n".join(f"reading {o}: {s}" for o, s in st.items()) or f"no {f}")
sys.exit(0 if st and all(s == "ok" for s in st.values()) else 1)
EOF
echo "OPPO-MC-60 finished: $(date), $FAILED failed step(s)"
[ "$FAILED" -eq 0 ]
