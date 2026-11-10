#!/usr/bin/env bash
# M5 follow-up (hns.marl.capability), under the simulator's current launderer reward (per proceeds): the root target
# at Q dropout 0 and 0.5 against two frozen M5 opponents (Full seed 0, NDA seed 1), plus the reference policies, a few
# processes at a time on the local GPU; then reports/m5_capability.md. Logs: runs/capability/proceeds/logs/.
# (The per-tranche runs, target root/mean x dropout 0/0.5, ran at commit 98369af: runs/capability/tranche/.)
#   scripts/m5_capability.sh [JOBS=2]   (one run peaks near 1.7 GB: 2 fit the local 4 GB GPU, 4 do not)
set -uo pipefail
cd "$(dirname "$0")/.."
LOGS=runs/capability/proceeds/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  for opp in full/0 nda/1; do for d in 0 0.5; do echo "train --opp runs/marl/$opp --target root --q-dropout $d"; done; done
  for opp in full/0 nda/1; do echo "refs --opp runs/marl/$opp"; done
} | LOGS=$LOGS xargs -P "${1:-2}" -I{} sh -c \
  'n=$(echo "{}" | tr " /" "__"); uv run python -m hns.marl.capability {} > "$LOGS/$n.log" 2>&1 || echo "FAILED: {}"'
uv run python -m hns.marl.capability report
