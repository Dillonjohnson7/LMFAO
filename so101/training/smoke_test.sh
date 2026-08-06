#!/usr/bin/env bash
# ============================================================================
# FREE local validation of the training config — run on the workcell BEFORE
# renting a GPU box (Mistake #11: a free CPU smoke test caught the missing
# [training] extra; institutionalized here). Trains a few steps on the LOCAL
# dataset with the exact flags train_act.sh uses, minus cuda.
#
#   bash smoke_test.sh                # 3 steps on datasets/pick_place_v2
#   STEPS=5 ROOT=... bash smoke_test.sh
#
# Takes ~1 min. If this passes, the cloud run's config is proven — the only
# untested pieces left are CUDA (gated by cloud_setup.sh) and the Hub pull.
# ============================================================================
set -euo pipefail

ROOT="${ROOT:-$HOME/SO101_policy/datasets/pick_place_v2}"
REPO="local/$(basename "$ROOT")"
STEPS="${STEPS:-3}"
AUG="${AUG:-1}"
OUT="${OUT:-/tmp/act_smoke_$$}"
PY=$HOME/SO101_policy/.venv

[ -f "$ROOT/meta/tasks.parquet" ] || { echo "no finalized dataset at $ROOT — record demos first (~/rec)"; exit 1; }

EXTRA=""
[ "$AUG" = "1" ] && EXTRA="--dataset.image_transforms.enable=true"

# shellcheck disable=SC2086
HF_HUB_OFFLINE=1 "$PY/bin/lerobot-train" \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --steps="$STEPS" \
  --log_freq=1 \
  --num_workers=0 \
  --save_freq="$STEPS" \
  --output_dir="$OUT" \
  --job_name=act_smoke \
  --wandb.enable=false \
  $EXTRA

echo
echo "SMOKE TEST PASSED — this exact config trains end-to-end ($STEPS steps, CPU)."
echo "(scratch output in $OUT — safe to delete)"
