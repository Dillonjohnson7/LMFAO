#!/usr/bin/env bash
# Train one ACT policy per evaluation bucket (stock + each LMFAO augmentation).
#
# Same hyperparameters across buckets so the comparison is about the data, not
# the trainer knobs. Requires CUDA + lerobot-train (see docs/STATUS.md §2).
#
# Usage:
#   OUT_ROOT=/data/eval_buckets ./scripts/train_eval_buckets.sh
#   BUCKETS="stock lighting noise" STEPS=50000 ./scripts/train_eval_buckets.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to the directory produced by augment_eval_buckets.sh}"
RUNS_ROOT="${RUNS_ROOT:-${LMFAO_DATA_ROOT:-/workspace/data}/runs/eval_buckets}"
BUCKETS="${BUCKETS:-stock lighting noise occlusion spatial full}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-7}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
LEROBOT_BIN="${LEROBOT_BIN:-lerobot-train}"
HF_USER="${HF_USER:-local}"

if ! command -v "$LEROBOT_BIN" >/dev/null 2>&1; then
  echo "cannot find ${LEROBOT_BIN} on PATH — install LeRobot 0.6.x first" >&2
  exit 1
fi
if [[ "$POLICY_DEVICE" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "POLICY_DEVICE=cuda but nvidia-smi is missing — run on a CUDA pod" >&2
  exit 1
fi

mkdir -p "$RUNS_ROOT"

# Validate the whole batch before starting any policy. LeRobot requires each
# output_dir to not exist when resume=false; pre-creating it makes training fail.
for bucket in $BUCKETS; do
  ds="${OUT_ROOT}/${bucket}"
  run="${RUNS_ROOT}/${bucket}"
  if [[ ! -d "$ds/meta" ]]; then
    echo "missing bucket dataset: ${ds} (run scripts/augment_eval_buckets.sh first)" >&2
    exit 1
  fi
  if [[ -e "$run" ]]; then
    echo "training output already exists: ${run}" >&2
    echo "choose a fresh RUNS_ROOT or move/remove that bucket explicitly" >&2
    exit 1
  fi
done

for bucket in $BUCKETS; do
  ds="${OUT_ROOT}/${bucket}"
  run="${RUNS_ROOT}/${bucket}"
  echo
  echo "=== train bucket=${bucket} steps=${STEPS} ==="
  "$LEROBOT_BIN" \
    --dataset.repo_id="${HF_USER}/lmfao_eval_${bucket}" \
    --dataset.root="$ds" \
    --policy.type=act \
    --policy.device="$POLICY_DEVICE" \
    --policy.push_to_hub=false \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --seed="$SEED" \
    --save_checkpoint="$SAVE_CHECKPOINT" \
    --wandb.enable="$WANDB_ENABLE" \
    --output_dir="$run"
done

echo
echo "Checkpoints under ${RUNS_ROOT}/{bucket}"
echo "Compare with physical rollouts — see docs/TRAINING_RUN.md §Compare"
