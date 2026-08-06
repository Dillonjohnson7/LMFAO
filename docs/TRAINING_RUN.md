# Eval run — recollect demos → augment buckets → train → compare

Goal: measure whether LMFAO's CLI augmentations actually improve policy
robustness. Not "train one big model on old HF data" — **recollect**, expand
with the CLI, train matched policies, compare on the robot.

## Experiment design

| bucket | training data | what it tests |
| --- | --- | --- |
| `stock` | freshly recorded demos only | baseline |
| `lighting` | stock + brightness/contrast/temperature | lighting robustness |
| `noise` | stock + gaussian/uniform | sensor noise robustness |
| `occlusion` | stock + box / border / moving occluders | partial view robustness |
| `spatial` | stock + random crop | framing / crop robustness |
| `full` | stock + combined ADJUST pipeline | all-effects cocktail |

Same ACT hyperparameters and seed across buckets. The only variable is the dataset.

Configs live under `configs/training/buckets/`. Scripts:

1. `scripts/record_demos.sh` / `so101/rec` — guided teleop on the SO101 workstation
2. `scripts/augment_eval_buckets.sh` — build each bucket with `lmfao augment`
3. `scripts/train_eval_buckets.sh` — one ACT checkpoint per bucket (CUDA pod)

Workcell tooling lives under [`so101/`](../so101/README.md) (from
`MultiplyLabor/SO101_policy`: guided recorder, camera pin, teleop, review).

## 1. Recollect new demos (robot workstation)

Do **not** reuse `Dillonjohnson/pick_place_v2` for this eval — that set was for
building/smoke-testing LMFAO. Record a new session under a controlled layout.

```bash
# On the SO101 machine (LeRobot 0.6.x env; NUC: LEROBOT_VENV=/home/multiply/envs/lerobot06):
# one-time: ./so101/setup.sh
# one-time cams: python so101/eval/cam_check.py --find-front && python so101/eval/cam_check.py --find
./scripts/record_demos.sh 40
# same as: ./so101/rec 40
```

Protocol:

- Fix lighting, camera mounting, and table scene for the whole session.
- Prefer slow, consistent teleop; at STEP 3 press ENTER to keep or `r` to redo.
- Optional: append a ~30s slow scene sweep after task episodes (future GENERATE
  hygiene; not required for ADJUST).
- Keep the camera set identical for record / train / rollout (front+wrist by
  default; `WRIST=0` for front-only).

Default dataset root: `so101/datasets/pick_place_v3`. Point `STOCK` at that
path (or a copy on a fast disk).

## 2. Augment with the CLI

CPU machine is fine:

```bash
pip install -e ".[dev,lerobot]"
export STOCK=/path/to/pick_place_v3          # freshly recorded root
export OUT_ROOT=/path/to/eval_buckets
./scripts/augment_eval_buckets.sh
# subset: BUCKETS="lighting noise" ./scripts/augment_eval_buckets.sh
```

Each bucket gets `--include-original` so training still sees the real demos plus
seasoned variants (`VARIANTS` default 8).

## 3. Train several policies

On a CUDA pod (LeRobot 0.6.x recipe in `docs/STATUS.md` §2):

```bash
export OUT_ROOT=/data/eval_buckets
export RUNS_ROOT=/data/runs/eval_buckets
export LEROBOT_BIN=/workspace/v312/bin/lerobot-train
./scripts/train_eval_buckets.sh

# smoke one bucket first:
BUCKETS=stock STEPS=10 BATCH_SIZE=2 SAVE_CHECKPOINT=false ./scripts/train_eval_buckets.sh
```

Transfer datasets with `COPYFILE_DISABLE=1 tar` or `hf upload` (avoid macOS
`._*` AppleDouble sidecars).

## 4. Compare how well the CLI helps

Loss curves are not the score. Compare **physical rollout success** under
held-out conditions the stock demos never saw:

| condition | example |
| --- | --- |
| lighting shift | room lights dimmed / warmer lamp |
| framing shift | camera nudged a few cm / slight zoom |
| occlusion | small object at image border |
| noise-ish | lower exposure / gain bump |

For each bucket checkpoint, run N trials (e.g. 20) of the same pick→place task
and log success / failure. Suggested summary table:

```text
bucket     | train lighting | held-out lighting | held-out framing
-----------|----------------|-------------------|-----------------
stock      |                |                   |
lighting   |                |                   |
noise      |                |                   |
occlusion  |                |                   |
spatial    |                |                   |
full       |                |                   |
```

A family "works" if its bucket beats `stock` on the matching held-out condition
without collapsing the in-distribution score.

Eval rollouts can reuse LeRobot recording with `--policy.path=...` into an
`eval_*` dataset (see upstream `lerobot-record` docs).

## Default knobs

| knob | default | notes |
| --- | --- | --- |
| `NUM_EPISODES` | 50 | recording session size |
| `VARIANTS` | 8 | seasoned copies per source episode |
| `STEPS` | 100000 | matched across buckets |
| `SEED` | 7 | shared augment + train seed |

## Out of scope for this eval

- Re-downloading / reusing `pick_place_v2` as the training set
- `lmfao generate` / Gaussian-splat novel views
- Diffusion policy
- Tuning augmenter magnitudes before the first stock-vs-buckets table exists
