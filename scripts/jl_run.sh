#!/usr/bin/env bash
# Start M5 on the instance as one managed run (returns at once). Watch it with
#   jl run logs <run_id> --tail 50      and      scripts/jl_status.sh
#   scripts/jl_run.sh [EPISODES=1000] [OUT=runs/marl] [RUNS="full:0 ... nda:2"]
set -euo pipefail
cd "$(dirname "$0")/.."
ID="$(cat runs/jl_machine_id)"
jl run --on "$ID" --json --yes --no-follow -- sh -lc "cd /home/hns && exec scripts/m5_all.sh '${1:-1000}' '${2:-runs/marl}' '${3:-full:0 full:1 full:2 nda:0 nda:1 nda:2}'"
