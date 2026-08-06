#!/usr/bin/env bash
# ============================================================================
# Train an ACT policy on the SO101 pick_place dataset (on a rented GPU box).
# Run after cloud_setup.sh:   bash train_act.sh   (self-locates the venv; no activation needed)
#
# V2 DEFAULTS (wrist-cam retrain, see V2_PIPELINE.md): two-camera dataset
# Dillonjohnson/pick_place_v2, 120k steps, image augmentation ON.
# The v1 run is reproducible with:  DATASET=Dillonjohnson/pick_place AUG=0 STEPS=150000
#
# The dataset is pulled from the private HF repo — nothing to copy; lerobot
# downloads it on first use. ALWAYS run training/smoke_test.sh on the workcell
# BEFORE renting the box (free CPU validation of this exact config).
# ============================================================================
set -euo pipefail

DATASET="${DATASET:-Dillonjohnson/pick_place_v2}"
# 120k, not 150k: the v1 plateau was measured flat past ~100k (slope -0.002 to
# -0.004 per 10k). Two 720p streams cost ~1.5-2x per step; past-plateau steps
# buy nothing (CHECKPOINT.md §2).
STEPS="${STEPS:-120000}"
# AUG=1 (default): lerobot's stock image transforms — brightness/contrast/
# saturation/hue/sharpness jitter + RandomAffine (±5°, 5% translate). Hedges
# the two DOCUMENTED recurring problems: camera-mount drift (twice) and
# lighting shift (night-vs-day, once). AUG=0 disables.
AUG="${AUG:-1}"
# NOVAE=1 -> train WITHOUT the CVAE (--policy.use_vae=false). This is an available
# ablation knob only. The original "the VAE latent memorized the actions and
# inference stayed vision-blind" rationale was DISPROVEN — the cube-removal test
# that seemed to prove it had run a randomly-initialized network, and the latent
# A/B test later showed the latent is healthy (see CHECKPOINT.md Mistakes #1-2).
# Do not reach for NOVAE=1 on that debunked basis.
EXTRA=""
[ "${NOVAE:-0}" = "1" ] && EXTRA="--policy.use_vae=false"
[ "$AUG" = "1" ] && EXTRA="$EXTRA --dataset.image_transforms.enable=true"
BATCH="${BATCH:-8}"          # ACT preset default. On a 24GB+ GPU you can raise to 32-64.
JOB="${JOB:-act_pick_place_v2}"
OUT="${OUT:-$HOME/outputs/train/$JOB}"
VENV="${VENV:-$HOME/lerobot-venv}"   # call the binary by absolute path — works with or without activation

# shellcheck disable=SC2086
"$VENV/bin/lerobot-train" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=8 \
  --output_dir="$OUT" \
  --job_name="$JOB" \
  --wandb.enable=false \
  $EXTRA

echo
echo "Training done. Checkpoints in: $OUT/checkpoints"
echo "The trainable policy for the arm is:  $OUT/checkpoints/last/pretrained_model"
echo
echo "*** THIS BOX IS EPHEMERAL — SAVE THE MODEL OFF-BOX NOW ***"
echo "  hf upload Dillonjohnson/act_pick_place \\"
echo "     $OUT/checkpoints/last/pretrained_model ckpt-v2-last --repo-type=model --private"
echo "  (ckpt-v2-last = the in-repo dir V2_PIPELINE.md §6 pulls from; omitting it"
echo "   would dump these files into the repo ROOT, on top of the old 20k model)"
