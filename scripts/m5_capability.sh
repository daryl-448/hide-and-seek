#!/usr/bin/env bash
# M5 follow-up (hns.marl.capability): the 2 x 2 launderer-learner variants (target root/mean x Q dropout 0/0.5)
# against two frozen M5 opponents (Full seed 0, NDA seed 1), plus the reference policies, a few processes at a
# time on the local GPU; then reports/m5_capability.md. Logs: runs/capability/logs/.
#   scripts/m5_capability.sh [JOBS=4]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/capability/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2
{
  for opp in full/0 nda/1; do echo "refs --opp runs/marl/$opp"; done
  for opp in full/0 nda/1; do for t in root mean; do for d in 0 0.5; do
    echo "train --opp runs/marl/$opp --target $t --q-dropout $d"
  done; done; done
} | xargs -P "${1:-4}" -I{} sh -c \
  'n=$(echo "{}" | tr " /" "__"); uv run python -m hns.marl.capability {} > "runs/capability/logs/$n.log" 2>&1 || echo "FAILED: {}"'
uv run python -m hns.marl.capability report
