#!/usr/bin/env bash
# Fetch a v3 policy from Hugging Face and prove it is comparable to stock.
#
#   ./so101/eval/load_policy.sh occlusion
#   ./so101/eval/load_policy.sh spatial
#
# The comparison is only meaningful if the policy under test differs from stock
# in exactly one way -- the dataset it was trained on. So this checks the
# execution-relevant config against stock and refuses on any mismatch:
# chunk_size and n_action_steps govern how much of a plan the arm commits to,
# and the feature shapes govern what the cameras must deliver. Normalisation
# statistics are expected to differ: each policy normalises to its own training
# corpus, and that difference IS the experiment.
set -euo pipefail

BUCKET="${1:?usage: load_policy.sh <bucket>   e.g. occlusion}"
SO101_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LMFAO_DIR="$(cd "$SO101_DIR/.." && pwd)"
DEST="$LMFAO_DIR/policies/$BUCKET"
REPO="Dillonjohnson/pick_place_v3_act_${BUCKET}"

# Retained SHA-256s from docs/policytraining_v3.md 12.x. A checkpoint that does
# not match the doc is not the artifact the experiment is about.
declare -A SHA=(
  [stock]=6718be1a97c1b644884d7038ee0d3cc611ae445a69cedd815506dc1e2ed68e0b
  [spatial]=5325ebc2af91cf565cc50325d0036cb626b97f224e63f1bae426defab38f248c
  [occlusion]=f476c84bf04df3daaa346a975ebed68781a0d54056751b742d0b09086c38dbe8
)

B=$'\033[1m'; G=$'\033[32m'; R=$'\033[31m'; X=$'\033[0m'
# shellcheck disable=SC1091
source "$SO101_DIR/env.sh"

python - <<'PY' || { echo "${R}not logged in to Hugging Face — run:  hf auth login${X}" >&2; exit 1; }
from huggingface_hub import get_token
import sys
sys.exit(0 if get_token() else 1)
PY

echo "${B}fetching${X} $REPO -> $DEST"
[[ -d "$DEST" ]] && { echo "$DEST already exists — remove it first" >&2; exit 1; }
hf download "$REPO" --local-dir "$DEST" >/dev/null

for f in model.safetensors config.json policy_preprocessor.json policy_postprocessor.json train_config.json; do
  [[ -f "$DEST/$f" ]] || { echo "${R}incomplete download: missing $f${X}" >&2; exit 1; }
done

if [[ -n "${SHA[$BUCKET]:-}" ]]; then
  got=$(sha256sum "$DEST/model.safetensors" | cut -d' ' -f1)
  if [[ "$got" != "${SHA[$BUCKET]}" ]]; then
    echo "${R}SHA-256 MISMATCH — this is not the retained checkpoint${X}" >&2
    echo "  expected ${SHA[$BUCKET]}" >&2
    echo "  got      $got" >&2
    exit 1
  fi
  echo "${G}SHA-256 matches the retained artifact${X}"
fi

echo
echo "${B}config parity vs stock${X}"
DEST="$DEST" STOCK="$LMFAO_DIR/policies/stock" python - <<'PY'
import json, os, sys
a = json.load(open(os.path.join(os.environ["STOCK"], "config.json")))
b = json.load(open(os.path.join(os.environ["DEST"], "config.json")))
bad = []
for k in ("type", "chunk_size", "n_action_steps", "n_obs_steps",
          "vision_backbone", "dim_model", "use_vae", "temporal_ensemble_coeff"):
    if a.get(k) != b.get(k):
        bad.append(f"  {k}: stock={a.get(k)}  this={b.get(k)}")
    else:
        print(f"  {k:24} {a.get(k)}")
for side in ("input_features", "output_features"):
    for k, v in a.get(side, {}).items():
        w = b.get(side, {}).get(k)
        if w != v:
            bad.append(f"  {side}.{k}: stock={v}  this={w}")
if bad:
    print("\nMISMATCH — not comparable to the stock run:")
    print("\n".join(bad))
    sys.exit(1)
print("\n  feature shapes identical to stock")
PY

echo
echo "${G}${B}$BUCKET ready${X} — matched to stock on everything except its training data."
echo "Run it the same way stock was run:"
echo "  DISPLAY_DATA=1 POLICY=$BUCKET ./so101/series 15"
