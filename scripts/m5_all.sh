#!/usr/bin/env bash
# Run M5's MARL runs side by side on one GPU (on the instance, from /home/hns). Each run logs to
# runs/marl/<arm>/<seed>/train.log, checkpoints every 20 episodes and resumes on restart, so this script
# can simply be started again after a pause or a preemption. Exits non-zero if any run failed.
#   scripts/m5_all.sh [EPISODES=1000] [OUT=runs/marl] [RUNS="full:0 full:1 full:2 nda:0 nda:1 nda:2"]
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/remote_env.sh ] && [ -d /home/uv ] && . scripts/remote_env.sh
EPISODES="${1:-1000}"; OUT="${2:-runs/marl}"; LIST="${3:-full:0 full:1 full:2 nda:0 nda:1 nda:2}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 POLARS_MAX_THREADS=2   # 6 processes share the CPUs
# On the rented instance only: CUDA MPS lets the processes share the GPU without time-slicing
# (measured on the A30, 6 runs: about 115 s per world with it, 240 s without). HNS_MPS=0 turns it off.
if [ -d /home/uv ] && [ "${HNS_MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null; then
  export CUDA_MPS_PIPE_DIRECTORY=/home/mps/pipe CUDA_MPS_LOG_DIRECTORY=/home/mps/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
fi
pids=()
for r in $LIST; do
  arm="${r%%:*}"; seed="${r##*:}"
  mkdir -p "$OUT/$arm/$seed"
  uv run python -m hns.marl.train configs/marl.yaml --arm "$arm" --seed "$seed" --episodes "$EPISODES" --out "$OUT" \
    >> "$OUT/$arm/$seed/train.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for r in $LIST; do tail -n 1 "$OUT/${r%%:*}/${r##*:}/train.log"; done
exit $fail
