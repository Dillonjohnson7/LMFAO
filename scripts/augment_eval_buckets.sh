#!/usr/bin/env bash
# Build the comparative augmentation buckets from a freshly recorded stock dataset.
#
# Buckets:
#   stock      — originals only (copied / passthrough inspect target)
#   lighting   — brightness + contrast + color temperature
#   noise      — gaussian + uniform
#   occlusion  — border / sequence / moving boxes
#   spatial    — random crop
#   full       — combined ADJUST pipeline
#
# Usage:
#   STOCK=/path/to/recorded_dataset ./scripts/augment_eval_buckets.sh
#   BUCKETS="lighting noise" STOCK=... ./scripts/augment_eval_buckets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOCK="${STOCK:?set STOCK to the freshly recorded LeRobot dataset root}"
OUT_ROOT="${OUT_ROOT:-${LMFAO_DATA_ROOT:-/workspace/data}/eval_buckets}"
BUCKETS="${BUCKETS:-lighting noise occlusion spatial full}"
# Originals are oversampled so each bucket trains mostly on the nominal
# distribution: COPIES/(COPIES+VARIANTS) = 2/3 -> 67% stock, 33% augmented.
# 2+1 (not 4+2) so each unique augmented variant gets ~2x the training
# repetition at the same ratio.
VARIANTS="${VARIANTS:-1}"
ORIGINAL_COPIES="${ORIGINAL_COPIES:-2}"
SEED="${SEED:-7}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/training/buckets}"

if [[ ! -d "$STOCK/meta" ]]; then
  echo "STOCK does not look like a LeRobot dataset: ${STOCK}" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# stock bucket = the recorded demos unchanged
STOCK_LINK="${OUT_ROOT}/stock"
if [[ -e "$STOCK_LINK" || -L "$STOCK_LINK" ]]; then
  rm -rf "$STOCK_LINK"
fi
ln -s "$(cd "$STOCK" && pwd)" "$STOCK_LINK"
echo "stock -> ${STOCK_LINK}"
lmfao inspect "$STOCK_LINK" --episodes | head -n 20 || true

for bucket in $BUCKETS; do
  cfg="${CONFIG_DIR}/${bucket}.json"
  out="${OUT_ROOT}/${bucket}"
  if [[ ! -f "$cfg" ]]; then
    echo "missing config for bucket '${bucket}': ${cfg}" >&2
    exit 1
  fi
  echo
  echo "=== bucket=${bucket} variants=${VARIANTS} original_copies=${ORIGINAL_COPIES} ==="
  lmfao augment \
    --input "$STOCK" \
    --output "$out" \
    --config "$cfg" \
    --variants "$VARIANTS" \
    --include-original \
    --original-copies "$ORIGINAL_COPIES" \
    --seed "$SEED" \
    --resume
  lmfao inspect "$out" --episodes | head -n 20 || true
done

echo
echo "Buckets ready under ${OUT_ROOT}"
echo "Next (CUDA pod): STOCK=${STOCK} OUT_ROOT=${OUT_ROOT} ./scripts/train_eval_buckets.sh"
