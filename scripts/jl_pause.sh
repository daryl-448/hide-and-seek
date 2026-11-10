#!/usr/bin/env bash
# End of session (SPEC §3): list what exists, then pause (default) or destroy our instance.
#   scripts/jl_pause.sh [pause|destroy]
set -euo pipefail
cd "$(dirname "$0")/.."
ID="$(cat runs/jl_machine_id)"
jl list --json
jl "${1:-pause}" "$ID" --yes --json
jl list --json
