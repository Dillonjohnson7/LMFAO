#!/usr/bin/env bash
# Train an ACT policy on an LMFAO-augmented LeRobot v3 dataset.
#
# Requires a CUDA machine with the LeRobot 0.6.x recipe from docs/STATUS.md:
#   uv venv --python 3.12 /workspace/v312
#   uv pip install --python /workspace/v312 \
#     "git+https://github.com/huggingface/lerobot.git" \
#     datasets "av>=15.0.0,<16.0.0" torchcodec accelerate
#
# Usage:
#   DATASET_ROOT=/data/pick_place_v2_augmented ./scripts/train_act.sh
#   # or source a env file:
#   set -a; source /path/to/train_act.env; set +a; ./scripts/train_act.sh
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${LMFAO_DATA_ROOT:-/workspace/data}/pick_place_v2_augmented}"
DATASET_REPO_ID="${DATASET_REPO_ID:-Dillonjohnson/pick_place_v2_augmented}"
OUTPUT_DIR="${OUTPUT_DIR:-${LMFAO_DATA_ROOT:-/workspace/data}/runs/act_pick_place_v2_aug}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-7}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
LEROBOT_BIN="${LEROBOT_BIN:-lerobot-train}"

if [[ ! -d "$DATASET_ROOT/meta" ]]; then
  echo "missing augmented dataset at ${DATASET_ROOT}; run scripts/augment_full.sh first" >&2
  exit 1
fi

if ! command -v "$LEROBOT_BIN" >/dev/null 2>&1; then
  echo "cannot find ${LEROBOT_BIN} on PATH" >&2
  echo "Install LeRobot 0.6.x per docs/STATUS.md §2, then re-run." >&2
  exit 1
fi

if [[ "$POLICY_DEVICE" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "POLICY_DEVICE=cuda but nvidia-smi is missing — this host has no GPU." >&2
  echo "Run this script on a CUDA pod (e.g. RunPod A4500/A6000)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Training ACT"
echo "  dataset=${DATASET_ROOT}"
echo "  output=${OUTPUT_DIR}"
echo "  steps=${STEPS} batch_size=${BATCH_SIZE} device=${POLICY_DEVICE} seed=${SEED}"

# Smoke path: STEPS=10 SAVE_CHECKPOINT=false for a quick load/train check.
exec "$LEROBOT_BIN" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device="$POLICY_DEVICE" \
  --policy.push_to_hub=false \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --seed="$SEED" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --wandb.enable="$WANDB_ENABLE" \
  --output_dir="$OUTPUT_DIR"
