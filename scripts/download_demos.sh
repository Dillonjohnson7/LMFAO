#!/usr/bin/env bash
# Download the canonical SO101 pick-and-place demonstrations.
#
# The HF dataset is gated. Export a token with read access before running:
#   export HF_TOKEN=hf_...     # do NOT commit or paste this into chat
#   ./scripts/download_demos.sh
set -euo pipefail

REPO_ID="${REPO_ID:-Dillonjohnson/pick_place_v2}"
OUT_DIR="${OUT_DIR:-${LMFAO_DATA_ROOT:-/workspace/data}/pick_place_v2}"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set." >&2
  echo "${REPO_ID} is a gated Hugging Face dataset — authenticate first:" >&2
  echo "  export HF_TOKEN=hf_...   # from https://huggingface.co/settings/tokens" >&2
  echo "  # accept access on https://huggingface.co/datasets/${REPO_ID} if prompted" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_DIR")"
echo "Downloading ${REPO_ID} -> ${OUT_DIR}"

python3 - <<PY
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="dataset",
    local_dir="${OUT_DIR}",
)
print(f"ready: {path}")
PY

echo
echo "Inspecting:"
lmfao inspect "$OUT_DIR" --episodes
