#!/usr/bin/env python
"""Pre-rollout gate: prove the ONLY thing differing between trials is the policy.

Every reference value here is derived from the training dataset and the policy
directory at run time. Nothing about the scene, the geometry, the pose or the
puck is written into this file -- stale constants are how ring_spot.py silently
rotted from v1 to v2 to v3, and how a harness quietly becomes the variable the
experiment thinks it is measuring.

    python so101/eval/preflight.py                       # stock, default dataset
    POLICY_DIR=.../policies/lighting python .../preflight.py
    DATASET_ROOT=... REPO=local/pick_place_v3 python .../preflight.py

Gates (all thresholds are in sigma of the demo distribution, not absolutes):
  weights    every tensor byte-equal to model.safetensors        (Mistake #1)
  cameras    live geometry+fps == the dataset's recorded streams
  holders    nothing else owns the arm or the cameras            (Mistake #15)
  pose       live joints inside the demos' start distribution
  lighting   live cast/brightness inside the demos' distribution
  puck       live placement inside the demos' placement region

Exit code is nonzero if any gate fails, so it can guard a scored series.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import torch

SO101 = Path(__file__).resolve().parents[1]
LMFAO = SO101.parent

POLICY_DIR = Path(os.environ.get("POLICY_DIR", LMFAO / "policies" / "stock"))
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", SO101 / "datasets" / "pick_place_v3"))
REPO = os.environ.get("REPO", f"local/{DATASET_ROOT.name}")
SIGMA_GATE = float(os.environ.get("SIGMA_GATE", "3"))   # how far off-distribution is too far
WARM_S = float(os.environ.get("WARM_S", "5"))           # past the camera AE/AWB knee
# Deviations below these cannot plausibly matter to the policy (the arm sweeps
# tens of degrees mid-task), so they bound the sigma test from below where the
# demo spread collapses to ~0. POSE_MAX_DEG is CHECKPOINT.md's absolute gate.
POSE_FLOOR_DEG = float(os.environ.get("POSE_FLOOR_DEG", "2.0"))
POSE_FLOOR_GRIP = float(os.environ.get("POSE_FLOOR_GRIP", "5.0"))
POSE_MAX_DEG = float(os.environ.get("POSE_MAX_DEG", "25"))
N_DEMO_FRAMES = int(os.environ.get("N_DEMO_FRAMES", "0"))  # 0 = every episode

G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[1m"; X = "\033[0m"
results = []


def gate(name, ok, detail):
    results.append((name, ok))
    mark = f"{G}PASS{X}" if ok else f"{R}FAIL{X}"
    print(f"  [{mark}] {B}{name:<9}{X} {detail}")


def zscore(v, mu, sd):
    return (v - mu) / sd if sd > 1e-9 else 0.0


# ---------------------------------------------------------------- dataset refs
info = json.loads((DATASET_ROOT / "meta" / "info.json").read_text())
cam_specs = {                                   # what the policy was TRAINED on
    k.split(".")[-1]: (v["shape"][1], v["shape"][0], v.get("info", {}).get("video.fps"))
    for k, v in info["features"].items() if k.startswith("observation.images.")
}
print(f"{B}PRE-ROLLOUT GATE{X}   policy={POLICY_DIR.name}  dataset={DATASET_ROOT.name}")
print(f"  references derived from {info['total_episodes']} episodes / {info['total_frames']} frames "
      f"@ {info['fps']} fps\n")

# ------------------------------------------------------------------- weights
try:
    from safetensors import safe_open
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(str(POLICY_DIR))
    cfg.device = "cpu"
    pol = get_policy_class(cfg.type).from_pretrained(str(POLICY_DIR), config=cfg).to("cpu").eval()
    sd = pol.state_dict()
    bad = n = 0
    params = 0
    with safe_open(str(POLICY_DIR / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k); m = sd.get(k)
            if m is None or m.shape != t.shape or not torch.equal(m.float(), t.float()):
                bad += 1
            else:
                n += 1; params += t.numel()
    gate("weights", bad == 0,
         f"{n} tensors / {params:,} params byte-equal" if bad == 0
         else f"{bad} tensors DIFFER — the loaded net is not this checkpoint")
    del pol, sd
except Exception as e:                                    # noqa: BLE001
    gate("weights", False, f"could not verify: {e}")

# ------------------------------------------------------------------- holders
try:
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=10).stdout.split()
    held = [x for x in names if any(t in x.lower() for t in ("so101", "anvil", "ros2", "camera"))]
    gate("holders", not held, "nothing else owns the arm/cameras" if not held
         else f"RUNNING: {held} — these hold the arm or the front cam")
except Exception:                                          # noqa: BLE001
    gate("holders", True, "docker not available (assuming nothing holds the devices)")

# ------------------------------------------------------------------- cameras
live_frames = {}
cam_ok = True
for role, (w, h, fps_want) in sorted(cam_specs.items()):
    pin = SO101 / f"{role}_cam.path"
    if not pin.exists():
        gate("cameras", False, f"{role}: not pinned ({pin.name} missing)"); cam_ok = False; continue
    dev = os.path.realpath(pin.read_text().strip())
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps_want or 30)
    t0 = time.perf_counter(); got = 0; frame = None
    while time.perf_counter() - t0 < WARM_S:               # warm past AE/AWB, then measure
        ok, f = cap.read()
        if ok: frame = f; got += 1
    t1 = time.perf_counter(); got2 = 0
    for _ in range(45):
        ok, f = cap.read()
        if ok: frame = f; got2 += 1
    fps = got2 / (time.perf_counter() - t1)
    cap.release()
    if frame is None:
        gate("cameras", False, f"{role}: no frames from {dev}"); cam_ok = False; continue
    live_frames[role] = frame
    gh, gw = frame.shape[:2]
    ok = (gw, gh) == (w, h) and fps > (fps_want or 30) * 0.9
    cam_ok &= ok
    gate("cameras", ok, f"{role}: {gw}x{gh} @ {fps:.1f} fps  (trained {w}x{h} @ {fps_want})")

# ---------------------------------------------------------------------- pose
try:
    import glob
    import pyarrow.parquet as pq
    starts = []
    for f in sorted(glob.glob(str(DATASET_ROOT / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(f).to_pandas()
        if "episode_index" not in t: continue
        for _ep, g in t.groupby("episode_index"):
            starts.append(np.asarray(g.sort_values("frame_index").iloc[0]["observation.state"], float))
    S = np.array(starts)

    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    MOTORS = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])}
    MOTORS["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
    calib_p = os.environ.get("FOLLOWER_CALIB", os.path.expanduser(
        "~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"))
    calib = {n: MotorCalibration(**c) for n, c in json.load(open(calib_p)).items()}
    bus = FeetechMotorsBus(port=os.environ.get("FOLLOWER_PORT", "/dev/so101_follower"),
                           motors=MOTORS, calibration=calib)
    bus.connect()                                    # read-only; torque untouched
    try:
        acc = {n: [] for n in MOTORS}
        for _ in range(30):
            for n, v in bus.sync_read("Present_Position", normalize=True, num_retry=3).items():
                acc[n].append(v)
            time.sleep(1 / 60)
    finally:
        bus.disconnect(disable_torque=False)
    live_pose = np.array([np.mean(acc[n]) for n in MOTORS])
    # Sigma alone is the wrong gate here: the recorder starts every demo from the
    # same home pose, so some joints have sd ~= 0 and a physically meaningless
    # 0.9 deg reads as 20 sigma. Each joint is allowed the larger of its demo
    # spread and a floor below which a deviation cannot matter, and is hard-capped
    # at CHECKPOINT.md's absolute >25 deg "do not roll" line regardless of spread.
    FLOOR = {n: (POSE_FLOOR_GRIP if n == "gripper" else POSE_FLOOR_DEG) for n in MOTORS}
    names = list(MOTORS)
    delta = np.array([live_pose[i] - S[:, i].mean() for i in range(len(names))])
    thresh = np.array([max(SIGMA_GATE * S[:, i].std(), FLOOR[names[i]]) for i in range(len(names))])
    thresh = np.minimum(thresh, POSE_MAX_DEG)
    ratio = np.abs(delta) / thresh
    worst = int(np.argmax(ratio))
    gate("pose", ratio.max() <= 1.0,
         f"worst {names[worst]} {live_pose[worst]:+.1f} vs demos "
         f"{S[:,worst].mean():+.1f}+/-{S[:,worst].std():.2f} "
         f"(off {delta[worst]:+.1f}, allowed +/-{thresh[worst]:.1f})")
except Exception as e:                                     # noqa: BLE001
    gate("pose", False, f"could not read follower: {e}")

# ----------------------------------------------------- lighting + puck (front)
try:
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO, root=str(DATASET_ROOT), video_backend="pyav")
    epi = ds.meta.episodes
    n_eps = ds.num_episodes
    idxs = range(n_eps) if N_DEMO_FRAMES <= 0 else np.linspace(0, n_eps - 1, N_DEMO_FRAMES).astype(int)

    def blue_blob(bgr, lo, hi):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cs:
            a = cv2.contourArea(c)
            if a < 60: continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r <= 0 or a / (np.pi * r * r) < 0.55: continue
            if best is None or a > best[0]: best = (a, x, y, r)
        return None if best is None else (best[1], best[2], best[3])

    # The puck's colour band is fitted to the demos rather than assumed: widen
    # until the detector explains nearly every demo start frame, then reuse that
    # exact band live. A band that cannot find the puck in the demos is not a
    # band worth grading the live scene with.
    demo0 = []
    for ep in idxs:
        i = int(epi["dataset_from_index"][int(ep)])
        im = ds[i]["observation.images.front"]
        demo0.append((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy())

    band = None
    for smin in (70, 50, 35, 25):
        lo, hi = (95, smin, 40), (135, 255, 255)
        hits = [blue_blob(f, lo, hi) for f in demo0]
        if sum(h is not None for h in hits) >= 0.9 * len(demo0):
            band = (lo, hi, hits); break
    if band is None:
        gate("puck", False, "no colour band explains the demo start frames — detector needs work")
        raise SystemExit
    lo, hi, hits = band
    P = np.array([h for h in hits if h is not None])
    cx, cy, rr = P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean()
    sx, sy = P[:, 0].std(), P[:, 1].std()
    px_cm = rr / 3.5                                  # ~7 cm puck, radius 3.5 cm

    br = np.array([f[..., 0].mean() / max(f[..., 2].mean(), 1e-6) for f in demo0])
    bright = np.array([f.mean() for f in demo0])

    front = live_frames.get("front")
    if front is None:
        gate("lighting", False, "no live front frame"); gate("puck", False, "no live front frame")
    else:
        lbr = front[..., 0].mean() / max(front[..., 2].mean(), 1e-6)
        lb = front.mean()
        zc, zb = zscore(lbr, br.mean(), br.std()), zscore(lb, bright.mean(), bright.std())
        gate("lighting", abs(zc) <= SIGMA_GATE and abs(zb) <= SIGMA_GATE,
             f"cast {lbr:.3f} vs {br.mean():.3f}+/-{br.std():.3f} ({zc:+.1f} sd) · "
             f"brightness {lb:.0f} vs {bright.mean():.0f} ({zb:+.1f} sd)")

        p = blue_blob(front, lo, hi)
        if p is None:
            gate("puck", False, "puck not found live — out of frame, occluded, or lighting shifted")
        else:
            lx, ly, lr = p
            dx, dy = (lx - cx) / px_cm, (ly - cy) / px_cm
            nsig = float(np.hypot(zscore(lx, cx, sx), zscore(ly, cy, sy)))
            gate("puck", nsig <= SIGMA_GATE,
                 f"offset {np.hypot(dx,dy):.2f} cm from demo mean ({nsig:.1f} sigma) · "
                 f"demo spread {sx/px_cm:.2f}x{sy/px_cm:.2f} cm · {px_cm:.1f} px/cm")
            out = SO101 / "eval" / "cam_probe" / "preflight_puck.jpg"
            vis = front.copy()
            cv2.circle(vis, (int(cx), int(cy)), int(rr), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), int(2 * max(sx, sy)), (0, 200, 200), 1)
            cv2.circle(vis, (int(lx), int(ly)), int(lr), (0, 0, 255), 2)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
            print(f"           overlay: {out}")
except SystemExit:
    pass
except Exception as e:                                     # noqa: BLE001
    gate("lighting", False, f"scene check failed: {e}")

failed = [n for n, ok in results if not ok]
print()
if failed:
    print(f"{R}{B}HOLD{X} — {', '.join(sorted(set(failed)))} outside the demos' distribution.")
    print("A trial run now measures the scene, not the policy.")
    sys.exit(1)
print(f"{G}{B}CLEAR{X} — scene, pose and policy all match the training distribution.")
