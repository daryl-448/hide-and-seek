#!/usr/bin/env bash
# Pull the MARL results: metrics, configs, logs and the final (models-only) checkpoints. The resume
# checkpoints (ckpt/last.pt: buffers, several hundred MB each) stay on the instance.
#   scripts/jl_pull.sh [OUT=runs/marl]
set -euo pipefail
cd "$(dirname "$0")/.."
ID="$(cat runs/jl_machine_id)"
OUT="${1:-runs/marl}"
jl exec "$ID" -- sh -lc "cd /home/hns && tar czf /home/hns_results.tgz --exclude='last.pt' --exclude='*.tmp' '$OUT'"
jl download "$ID" /home/hns_results.tgz runs/hns_results.tgz
tar xzf runs/hns_results.tgz && rm runs/hns_results.tgz
find "$OUT" -name result.json | sort
