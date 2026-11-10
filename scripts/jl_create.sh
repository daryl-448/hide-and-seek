#!/usr/bin/env bash
# Create the project's JarvisLabs GPU container (SPEC §3: one A30, 24 GB, region IN2, 100 GB).
# Prints the machine id and stores it in runs/jl_machine_id (git-ignored). Billing starts here:
# pause it with scripts/jl_pause.sh when the session ends.
#   scripts/jl_create.sh [GPU=A30] [--spot]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-A30}"; shift || true
jl gpus --json | python3 -c '
import json, sys
gpu = sys.argv[1]
rows = [r for r in json.load(sys.stdin) if r["gpu_type"] == gpu and r.get("workload_type") in ("container", None)]
free = max((r["effective_num_free_devices"] for r in rows), default=0)
price = rows[0]["price_per_hour"] if rows else None
print(f"{gpu}: {free} free, {price} USD/h on-demand" if rows else f"{gpu}: not offered")
sys.exit(0 if free > 0 else 1)' "$GPU" || { echo "no free $GPU: see SPEC §16 (an equivalent >= 24 GB at <= \$0.6/h; note it in the run config)" >&2; exit 1; }
out="$(jl create --gpu "$GPU" --region IN2 --storage 100 --name hns --yes --json "$@")"
id="$(printf '%s' "$out" | python3 -c 'import json, sys; print(json.load(sys.stdin)["machine_id"])')"
mkdir -p runs && printf '%s\n' "$id" > runs/jl_machine_id
echo "machine $id"
