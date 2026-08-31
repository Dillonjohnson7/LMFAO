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
# ---- EPOCHS mode: hold passes-over-the-data constant across dataset sizes ----
# The stock-vs-augmented eval (76% vs 8%, Aug 2026) was confounded: fixed STEPS
# over a 3x-bigger augmented dataset trained each sample 1/3 as often. Per
# Andrew's guidance: with model size fixed, keep the chinchilla-style ratio by
# holding EPOCHS (total pass-throughs) constant-or-growing, i.e. STEPS must
# scale with dataset size. Set EPOCHS to derive STEPS from the dataset itself:
#   EPOCHS=34 DATASET=... bash train.sh     # 34 ~= the stock baseline
#     (ACT 100k steps x batch 8 = 800k samples / 23,233 frames = 34.4 epochs)
# EPOCHS overrides STEPS. Works for local roots and HF repo ids.
if [ -n "${EPOCHS:-}" ]; then
  FRAMES=$(python3 - "$DATASET_ROOT" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
p = os.path.join(d, "meta", "info.json")
if os.path.exists(p):
    info = json.load(open(p))
else:
    from huggingface_hub import hf_hub_download
    info = json.load(open(hf_hub_download(d, "meta/info.json", repo_type="dataset")))
print(info["total_frames"])
PYEOF
)
  STEPS=$(( (EPOCHS * FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
  echo "EPOCHS=$EPOCHS over $FRAMES frames @ batch $BATCH_SIZE -> STEPS=$STEPS"
fi

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
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "training output already exists: ${OUTPUT_DIR}" >&2
  echo "choose a fresh OUTPUT_DIR or move/remove it explicitly" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
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
