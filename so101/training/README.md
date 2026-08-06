# SO101 policy training — cloud runbook

Train imitation-learning policies on the `pick_place` dataset collected on the
SO101 workcell. **This workcell has no GPU** — training runs on a rented cloud
GPU; this box only records data and (later) runs the trained policy on the arm.

## The dataset (already on the Hub)

- **Repo:** `Dillonjohnson/pick_place` — **private** HF dataset, revision tag **`v3.0`**
- 30 episodes · 21,186 frames · 30 fps · `robot_type: so_follower`
- One camera `observation.images.front` (1280×720, AV1 video) · 6-DOF action + state
- Uploaded from `~/SO101_policy/datasets/pick_place` (canonical `meta/`+`data/`+`videos/` only;
  the 4.1 GB `images/` recording-scratch was excluded — training decodes the video).
- To re-push after collecting more episodes: `python <scratch>/push_dataset.py`
  (or `hf upload Dillonjohnson/pick_place <root> --repo-type=dataset` + re-tag).

## Files here

| file | what |
|---|---|
| `cloud_setup.sh` | Bootstrap a fresh rented GPU box (uv + pinned lerobot + HF login). |
| `train_act.sh`   | The ACT training run, wired to the Hub dataset. |
| `README.md`      | This runbook. |

## Steps

### 1. Rent a GPU box
Any of RunPod / Lambda / Vast / Paperspace. Recommended for ACT:
- **1× GPU**, 24 GB is plenty (ACT + resnet18 is a ~52M-param model — needs only a few GB).
- **8+ CPU cores** (the dataloader decodes AV1 video frames on CPU).
- ~20 GB disk. Pick an image that already has NVIDIA driver + CUDA (e.g.
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).

**Cheapest RunPod GPUs with enough VRAM (surveyed 2026-07-17, on-demand):**

| GPU | VRAM | $/hr | notes |
|---|---|---|---|
| **RTX A5000** | 24 GB | **$0.16** | best value; community + secure |
| RTX A4500 | 20 GB | $0.19 | |
| RTX 3090 | 24 GB | $0.22 | |
| RTX 4090 | 24 GB | $0.34 | fastest 24 GB, but community capacity is flaky |
| A100 80 GB | 80 GB | $1.39 | overkill for ACT; reliable/available on Secure |

- **Prefer Secure Cloud** for reliability: we hit a **bad community 4090** whose GPU
  passed `nvidia-smi` but failed CUDA compute (error 999). Always **health-check CUDA
  before installing** (see `cloud_setup.sh` step 5 — it asserts `torch.cuda.is_available()`).
- The driver must support **CUDA 12.8** (our pinned `torch` is `+cu128`). Secure hosts
  with driver ≥ 570 are fine; the assert catches any mismatch cheaply.
- **Default recommendation: RTX A5000 on Secure Cloud ($0.16/hr).**

### 2. Bootstrap (once per box)
Copy `cloud_setup.sh` + `train_act.sh` to the box, then:
```bash
bash cloud_setup.sh      # installs deps, pins lerobot, checks CUDA, hf auth login
```

### 3. Train
```bash
source ~/lerobot-venv/bin/activate
bash train_act.sh        # pulls Dillonjohnson/pick_place, trains ACT
```
- **~100k steps, batch 8.** On a 4090/A100 expect **~a few hours** wall-clock
  (dataloading-bound more than compute-bound).
- No eval loop runs: SO101 has no sim env, so evaluation is on the **real arm**
  back at the workcell (next phase, not part of this run).
- Tune via env vars: `STEPS=`, `BATCH=` (raise to 32–64 on a big GPU), `DATASET=`.

### 4. ⚠ SAVE THE MODEL OFF-BOX — a rented box is wiped on teardown
The trained policy is `outputs/train/act_pick_place/checkpoints/last/pretrained_model`.
Push it to the Hub before you kill the box:
```bash
hf upload Dillonjohnson/act_pick_place \
  ~/outputs/train/act_pick_place/checkpoints/last/pretrained_model \
  --repo-type=model --private
```

### 5. (Later) Run it on the arm
Back on the workcell, pull the model and drive the follower with lerobot's record/eval
path (real-robot inference). That's a separate session — flag it when you're ready and
we'll wire up the on-arm eval + safety (the arm must start at rest; see the workcell notes).

## Why lerobot is pinned to a git commit (not PyPI)
The dataset is `codebase_version v3.0`. The workcell runs lerobot **0.6.1**, which is
**not published on PyPI** (PyPI stops at 0.6.0). To read v3.0 identically, the cloud
installs the exact commit the workcell uses:
`git+https://github.com/huggingface/lerobot.git@3f2179f3b69708b6ad009b2e7685dd9d05269ee1`.

## pi0 later
Same box, same dataset. pi0 is a pretrained VLA: heavier GPU (want 40–80 GB), longer
finetune, and it benefits from more than 30 demos — so get ACT working and eval'd on the
arm first, then revisit pi0 with `--policy.type=pi0` (and its extra deps).
