#!/usr/bin/env bash
# Last log line of every MARL run on the instance, and the GPU's load.
#   scripts/jl_status.sh [OUT=runs/marl]
set -euo pipefail
cd "$(dirname "$0")/.."
ID="$(cat runs/jl_machine_id)"
jl exec "$ID" -- sh -lc "cd /home/hns && for f in ${1:-runs/marl}/*/*/train.log; do printf '%s: ' \"\$f\"; grep -v -i warn \"\$f\" | tail -n 1; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime"
