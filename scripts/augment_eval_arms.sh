#!/usr/bin/env bash
# Build the comparative augmentation arms from a freshly recorded stock dataset.
#
# Arms:
#   stock      — originals only (copied / passthrough inspect target)
#   lighting   — brightness + contrast + color temperature
#   noise      — gaussian + uniform
#   occlusion  — border / sequence / moving boxes
#   spatial    — random crop
#   full       — combined ADJUST pipeline
#
# Usage:
#   STOCK=/path/to/recorded_dataset ./scripts/augment_eval_arms.sh
#   ARMS="lighting noise" STOCK=... ./scripts/augment_eval_arms.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOCK="${STOCK:?set STOCK to the freshly recorded LeRobot dataset root}"
OUT_ROOT="${OUT_ROOT:-${LMFAO_DATA_ROOT:-/workspace/data}/eval_arms}"
ARMS="${ARMS:-lighting noise occlusion spatial full}"
VARIANTS="${VARIANTS:-8}"
SEED="${SEED:-7}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/training/arms}"

if [[ ! -d "$STOCK/meta" ]]; then
  echo "STOCK does not look like a LeRobot dataset: ${STOCK}" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# stock arm = the recorded demos unchanged
STOCK_LINK="${OUT_ROOT}/stock"
if [[ -e "$STOCK_LINK" || -L "$STOCK_LINK" ]]; then
  rm -rf "$STOCK_LINK"
fi
ln -s "$(cd "$STOCK" && pwd)" "$STOCK_LINK"
echo "stock -> ${STOCK_LINK}"
lmfao inspect "$STOCK_LINK" --episodes | head -n 20 || true

for arm in $ARMS; do
  cfg="${CONFIG_DIR}/${arm}.json"
  out="${OUT_ROOT}/${arm}"
  if [[ ! -f "$cfg" ]]; then
    echo "missing config for arm '${arm}': ${cfg}" >&2
    exit 1
  fi
  echo
  echo "=== arm=${arm} variants=${VARIANTS} ==="
  lmfao augment \
    --input "$STOCK" \
    --output "$out" \
    --config "$cfg" \
    --variants "$VARIANTS" \
    --include-original \
    --seed "$SEED" \
    --resume
  lmfao inspect "$out" --episodes | head -n 20 || true
done

echo
echo "Arms ready under ${OUT_ROOT}"
echo "Next (CUDA pod): STOCK=${STOCK} OUT_ROOT=${OUT_ROOT} ./scripts/train_eval_arms.sh"
