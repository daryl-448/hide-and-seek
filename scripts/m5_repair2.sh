#!/usr/bin/env bash
# M5 repair cycle 2 (hns.marl.repair2; declared in its docstring and SPEC §10): the screen. For each frozen opponent
# (Full seed 0, NDA seed 1): the fixed diagnostic panel and the reference policies on the selection worlds, then the
# 12 cells in the declared order, two processes at a time on the local GPU (one peaks near 1.7 GB of its 4 GB).
# A finished run is skipped, a killed one starts again. Then the selection and reports/m5_repair2.md.
#   scripts/m5_repair2.sh [JOBS=2]      logs: runs/repair2/logs/
# Before the screen:  scripts/m5_repair2.sh stage0   (cell A against both opponents, then its timeline; about 45 min)
# After a selection:  scripts/m5_repair2.sh confirm [JOBS=2]   (seeds 1 and 2, the confirmation worlds, the verdict)
# Stage 1 (hns.marl.stage1): scripts/m5_repair2.sh stage1 CELL OPP   (one run and its reading, 25-minute GPU cap;
#   the blocks before PLR1 had 45 minutes)
# OPPO-MC (hns.marl.oppo_block): scripts/m5_repair2.sh oppo OPP   (one block: train and read, 45-minute cap)
set -uo pipefail
cd "$(dirname "$0")/.."
LOGS=runs/repair2/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # stdin: one repair2 command line per job
  LOGS=$LOGS xargs -P "$1" -I{} sh -c \
    'n=$(echo "{}" | tr " /" "__"); uv run python -m hns.marl.repair2 {} >> "$LOGS/$n.log" 2>&1 || echo "FAILED: {}"'
}
if [ "${1:-}" = stage0 ]; then
  for o in full0 nda1; do [ -f runs/repair2/panel_$o.pkl ] || echo "panel --opp $o"; done | run 2
  for o in full0 nda1; do echo "train --cell A --opp $o"; done | run 2
  for o in full0 nda1; do echo "timeline --cell A --opp $o"; done | run 2
  exit
fi
if [ "${1:-}" = stage1 ]; then          # one block, one attempt: train, then read worlds 20 and 30 (stage1 run)
  timeout 1500 uv run python -m hns.marl.stage1 run --cell "$2" --opp "$3" >> "$LOGS/stage1_$2_$3.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || echo "stage 1 $2 $3: exit $rc (124 = cut by the 25-minute cap); the block is incomplete and is not run again"
  exit $rc
fi
if [ "${1:-}" = oppo ]; then           # OPPO-MC (hns.marl.oppo_block): one block, one attempt, 45-minute cap
  timeout 2700 uv run python -m hns.marl.oppo_block run --opp "$2" >> "$LOGS/oppo_$2.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || echo "OPPO-MC $2: exit $rc (124 = cut by the 45-minute cap); the block is incomplete and is not run again"
  exit $rc
fi
if [ "${1:-}" = confirm ]; then
  J=${2:-2}
  CELL=$(uv run python -c "import json; print(json.load(open('runs/repair2/selection.json'))['cell'] or '')")
  [ -n "$CELL" ] || { echo "no cell was selected: the cycle has failed"; exit 1; }
  { for s in 1 2; do for o in full0 nda1; do echo "train --cell $CELL --opp $o --seed $s"; done; done
    for o in full0 nda1; do echo "refs --opp $o --worlds confirmation"; done; } | run "$J"
  for s in 0 1 2; do for o in full0 nda1; do echo "confirm-eval --opp $o --seed $s"; done; done | run "$J"
  for o in full0 nda1; do for c in "$CELL" A; do echo "sweep --cell $c --opp $o"; done; done | run "$J"
  uv run python -m hns.marl.repair2 confirm
else
  J=${1:-2}
  for o in full0 nda1; do [ -f runs/repair2/panel_$o.pkl ] || echo "panel --opp $o"; done | run "$J"
  for o in full0 nda1; do echo "refs --opp $o --worlds selection"; done | run "$J"
  for c in A SSF IHF IHR SHF ISF SSR SHR ISR W H LN; do for o in full0 nda1; do echo "train --cell $c --opp $o"; done; done | run "$J"
  uv run python -m hns.marl.repair2 select
fi
uv run python -m hns.marl.repair2 report
