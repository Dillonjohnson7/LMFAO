#!/usr/bin/env bash
# Train a single ACT policy on one dataset (used by train_eval_buckets.sh
# buckets, or for ad-hoc one-offs). Prefer scripts/train_eval_buckets.sh for
# the CLI evaluation experiment.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/lmfao_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${LMFAO_DATA_ROOT:-/workspace/data}/runs/act_single}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-7}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
LEROBOT_BIN="${LEROBOT_BIN:-lerobot-train}"

if [[ ! -d "$DATASET_ROOT/meta" ]]; then
  echo "missing dataset at ${DATASET_ROOT}" >&2
  exit 1
fi
if ! command -v "$LEROBOT_BIN" >/dev/null 2>&1; then
  echo "cannot find ${LEROBOT_BIN} on PATH" >&2
  exit 1
fi
if [[ "$POLICY_DEVICE" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "POLICY_DEVICE=cuda but nvidia-smi is missing — run on a CUDA pod" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
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
