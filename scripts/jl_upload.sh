#!/usr/bin/env bash
# Upload the code and the processed data to the instance and build the uv env under /home (the only
# path that survives a pause). Never the raw CSVs (SPEC E13, E14): only data/{parquet,graph,splits,norm}.
#   scripts/jl_upload.sh [--code-only]
set -euo pipefail
cd "$(dirname "$0")/.."
ID="$(cat runs/jl_machine_id)"
TMP="$(mktemp -d -p data jl.XXXXXX)"      # scratch under data/, not /tmp (SPEC §3)
trap 'rm -rf "$TMP"' EXIT
# tracked files plus new, non-ignored ones: the ignore rules keep dataset/, data/, runs/ and the PDFs out
git ls-files -co --exclude-standard -z | tar czf "$TMP/hns_code.tgz" --null -T -
# the instance has no git: record HEAD, and for an unclean tree a hash of the uploaded files
SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  SHA="$SHA-dirty.$(git ls-files -co --exclude-standard -z | xargs -0 sha1sum | sha1sum | cut -c1-12)"
fi
printf '%s\n' "$SHA" > "$TMP/.git_sha"
jl upload "$ID" "$TMP/.git_sha" /home/hns_git_sha
jl upload "$ID" "$TMP/hns_code.tgz" /home/hns_code.tgz
if [ "${1:-}" != "--code-only" ]; then
  tar cf "$TMP/hns_data.tar" data/parquet data/graph data/splits data/norm
  jl upload "$ID" "$TMP/hns_data.tar" /home/hns_data.tar
fi
jl exec "$ID" -- sh -lc '
set -e
mkdir -p /home/hns && cd /home/hns
tar xzf /home/hns_code.tgz
cp /home/hns_git_sha .git_sha
if [ -f /home/hns_data.tar ]; then tar xf /home/hns_data.tar && rm /home/hns_data.tar; fi
. scripts/remote_env.sh
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python -c "import torch; a = torch.cuda.get_arch_list(); assert \"sm_80\" in a and \"sm_75\" in a, a; print(torch.__version__, torch.cuda.get_device_name(0))"
'
