#!/usr/bin/env python
"""
OFFLINE eval of the trained ACT policy — NO ARM, NO CAMERA needed.

Replays the recorded dataset through the policy and compares the policy's
predicted actions against the human demonstration actions, per joint. This is
the safe "does the model actually work" check to run BEFORE moving the real arm.

  - Loads the model from the local snapshot (default) or the Hub repo.
  - Uses lerobot's real preprocessor/postprocessor (correct normalization).
  - Runs closed-loop-ish over an episode: select_action() re-infers every
    n_action_steps (=100) just like deployment, so the numbers reflect reality.

The dataset is stored in episode order, so this evaluates the first n_frames
(the start of episode 0). Cheap on purpose — it does not scan/decode all 21k
frames to seek an arbitrary episode.

Usage:  python offline_eval.py [n_frames]
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

MODEL = os.environ.get("MODEL", "/home/anvil/SO101_policy/models/act_pick_place")
ROOT = os.environ.get("DATASET_ROOT", "/home/anvil/SO101_policy/datasets/pick_place")
REPO = os.environ.get("REPO", "local/pick_place")  # on-disk data was created under this id; fast local load
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 250

print(f"Loading dataset ({REPO}) ...", flush=True)
ds = LeRobotDataset(REPO, root=ROOT, video_backend="pyav")

# The dataset is stored in episode order, so the first NMAX frames ARE the start
# of episode 0 — no need to scan/decode all 21k frames.
start = 0
end = min(NMAX, len(ds))
print(f"Evaluating the first {end} frames of the dataset (start of episode 0) ...", flush=True)

print(f"Loading policy from {MODEL} ...", flush=True)
cfg = PreTrainedConfig.from_pretrained(MODEL)
cfg.device = "cpu"
policy = ACTPolicy.from_pretrained(MODEL); policy.to("cpu")
policy.eval()
# Optional: force per-step re-inference (closed-loop-like) instead of the 100-step
# open-loop chunk. NSTEP=1 = re-infer every tick (fairest imitation accuracy).
NSTEP = int(os.environ.get("NSTEP", "0"))
if NSTEP > 0:
    try:
        policy.config.n_action_steps = NSTEP
        print(f"(forcing n_action_steps={NSTEP} — closed-loop-style eval)", flush=True)
    except Exception as e:
        print("could not set n_action_steps:", e, flush=True)
# The saved processors were written on the A100 with device=cuda; force cpu here.
_dev = {"device_processor": {"device": "cpu"}}
pre, post = make_pre_post_processors(cfg, pretrained_path=MODEL, dataset_stats=ds.meta.stats,
                                     preprocessor_overrides=_dev, postprocessor_overrides=_dev)

import time
policy.reset()
errs = []
t0 = time.time()
with torch.no_grad():
    for n, i in enumerate(range(start, end)):
        s = ds[i]
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
        batch["task"] = s.get("task", "")
        obs = pre(batch)
        action = policy.select_action(obs)
        action = post(action)
        a = action.squeeze(0).float().cpu().numpy()
        g = s["action"].float().cpu().numpy()
        errs.append(np.abs(a - g))
        if n % 20 == 0:
            print(f"  frame {n+1}/{end-start}  ({(time.time()-t0)/(n+1):.2f}s/frame)", flush=True)

errs = np.array(errs)  # (T, 6)
print("\n=== per-joint action error (degrees; gripper is 0-100) ===", flush=True)
print(f"{'joint':<15} {'MAE':>8} {'max':>8}")
for j, name in enumerate(JOINTS):
    print(f"{name:<15} {errs[:,j].mean():>8.2f} {errs[:,j].max():>8.2f}")
print(f"{'OVERALL':<15} {errs.mean():>8.2f} {errs.max():>8.2f}")
print(f"\nInterpret: low MAE (a few degrees) = the policy closely reproduces the demos.")
print(f"This is imitation accuracy, NOT task success — the real test is the arm.")
