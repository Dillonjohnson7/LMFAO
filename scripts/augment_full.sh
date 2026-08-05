#!/usr/bin/env bash
# Full-scale ADJUST augmentation over pick_place_v2 (no --limit / --max-frames).
# Produces a LeRobot-trainable dataset for ACT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${LMFAO_DATA_ROOT:-/workspace/data}"
INPUT="${INPUT:-${DATA_ROOT}/pick_place_v2}"
OUTPUT="${OUTPUT:-${DATA_ROOT}/pick_place_v2_augmented}"
CONFIG="${CONFIG:-${ROOT}/configs/training/pick_place_v2_pipeline.json}"
VARIANTS="${VARIANTS:-14}"
SEED="${SEED:-7}"

if [[ ! -d "$INPUT/meta" ]]; then
  echo "missing input dataset at ${INPUT}; run scripts/download_demos.sh first" >&2
  exit 1
fi

echo "Augmenting ${INPUT}"
echo "  -> ${OUTPUT}"
echo "  config=${CONFIG} variants=${VARIANTS} seed=${SEED}"

lmfao augment \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --config "$CONFIG" \
  --variants "$VARIANTS" \
  --include-original \
  --seed "$SEED" \
  --resume

echo
echo "Inspecting output:"
lmfao inspect "$OUTPUT" --episodes
echo
echo "Augmented dataset ready at ${OUTPUT}"
echo "Next (on a CUDA pod): scripts/train_act.sh"
