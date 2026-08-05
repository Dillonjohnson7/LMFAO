#!/usr/bin/env bash
# Train one ACT policy per evaluation arm (stock + each LMFAO augmentation).
#
# Same hyperparameters across arms so the comparison is about the data, not the
# trainer knobs. Requires CUDA + lerobot-train (see docs/STATUS.md §2).
#
# Usage:
#   OUT_ROOT=/data/eval_arms ./scripts/train_eval_arms.sh
#   ARMS="stock lighting noise" STEPS=50000 ./scripts/train_eval_arms.sh
set -euo pipefail

OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to the directory produced by augment_eval_arms.sh}"
RUNS_ROOT="${RUNS_ROOT:-${LMFAO_DATA_ROOT:-/workspace/data}/runs/eval_arms}"
ARMS="${ARMS:-stock lighting noise occlusion spatial full}"
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

for arm in $ARMS; do
  ds="${OUT_ROOT}/${arm}"
  run="${RUNS_ROOT}/${arm}"
  if [[ ! -d "$ds/meta" ]]; then
    echo "missing arm dataset: ${ds} (run scripts/augment_eval_arms.sh first)" >&2
    exit 1
  fi
  mkdir -p "$run"
  echo
  echo "=== train arm=${arm} steps=${STEPS} ==="
  "$LEROBOT_BIN" \
    --dataset.repo_id="${HF_USER}/lmfao_eval_${arm}" \
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
echo "Checkpoints under ${RUNS_ROOT}/{arm}"
echo "Compare with physical rollouts — see docs/TRAINING_RUN.md §Compare"
