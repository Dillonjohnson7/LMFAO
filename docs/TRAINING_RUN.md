# Training run — pick_place_v2 → augment → ACT

Operational checklist for the first full-scale ACT run on LMFAO-augmented
`Dillonjohnson/pick_place_v2`. Complements `docs/STATUS.md` §4B.

## What's already proven

- `lmfao augment` is production-ready at native resolution (streaming + `--resume`).
- Augmented LeRobot v3 output loads in lerobot 0.6.2.
- ACT smoke test on RunPod (RTX A4500): loss 74 → 22 over 10 steps.

## What this run does

1. **Collect demos** — use the canonical HF dataset (already recorded on SO101 /
   wrist cam). No new teleop in this repo.
2. **Expand demos** — season every episode with lighting / noise / occlusion /
   crop variants (`configs/training/pick_place_v2_pipeline.json`).
3. **Train policy** — ACT via upstream `lerobot-train` on a CUDA pod.

## Commands

```bash
# 0. Install LMFAO (CPU machine is fine for steps 1–2)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,lerobot]" huggingface_hub

# 1. Download demonstrations (gated HF dataset — token required)
export LMFAO_DATA_ROOT=/workspace/data   # or any non-iCloud path
export HF_TOKEN=hf_...                   # set locally; never commit / paste into chat
./scripts/download_demos.sh

# 2. Full augmentation (all episodes, 14 variants + originals, resume-safe)
./scripts/augment_full.sh
# Quick sanity (optional): VARIANTS=2 INPUT=... OUTPUT=.../smoke ./scripts/augment_full.sh

# 3. On a CUDA pod — install LeRobot 0.6.x (Python 3.12, av pinned <16)
uv venv --python 3.12 /workspace/v312
uv pip install --python /workspace/v312 \
  "git+https://github.com/huggingface/lerobot.git" \
  datasets "av>=15.0.0,<16.0.0" torchcodec accelerate

# Transfer the augmented dataset (avoid macOS AppleDouble sidecars):
#   COPYFILE_DISABLE=1 tar ...   OR   huggingface-cli upload ...

# 4. Train ACT
export DATASET_ROOT=/data/pick_place_v2_augmented
export LEROBOT_BIN=/workspace/v312/bin/lerobot-train
./scripts/train_act.sh

# Smoke (10 steps, no checkpoint):
STEPS=10 BATCH_SIZE=2 SAVE_CHECKPOINT=false ./scripts/train_act.sh
```

## Default experiment knobs

| knob | default | notes |
| --- | --- | --- |
| variants | 14 | per source episode, plus `--include-original` |
| seed | 7 | shared by augment + train |
| ACT steps | 100000 | override with `STEPS=` |
| batch size | 8 | smoke used 2 on A4500 |

## Success criteria (beyond loss)

Loss alone is not enough. Before calling the run done:

1. Hold out a lighting / framing condition not seen in training.
2. Define a physical rollout success metric on the SO101 (e.g. pick→place
   success over N trials).
3. Compare at least two datasets: originals-only vs originals+augmented.

## Out of scope here

- New teleop recording (use LeRobot's `lerobot-record` on the robot workstation).
- `lmfao generate` / Gaussian-splat novel views (experimental; not training-grade).
- Diffusion policy.
