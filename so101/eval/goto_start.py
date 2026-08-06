#!/usr/bin/env python
"""Drive the follower to the demo-mean start pose, so every trial begins alike.

    python so101/eval/goto_start.py            # plan, confirm, then move
    python so101/eval/goto_start.py --dry      # plan only; never energises
    python so101/eval/goto_start.py --yes      # skip the prompt (scored series)

The follower goes limp on disconnect and sags under gravity -- 5.4 deg at
shoulder_lift between two rollouts, measured -- so without this each trial
starts wherever the last one left the arm. That is several cm at the gripper
against a ~1 cm grasp budget: variance that is not the policy under test.

THE ARM MOVES (unless --dry). Clear the swept volume. Ctrl-C stops and holds.

Target and limits are read from the training dataset on every run, never
written into this file. creep_test.py still carries v2's start pose
(shoulder_pan -40.1, where v3's demos mean -2.1 -- 38 deg away): driving to a
stale constant is exactly the failure this is built to avoid.

Safety, following CHECKPOINT.md's guardrail stack:
  * the whole plan is computed from a READ-ONLY bus; torque is only touched
    once the journey has been checked and confirmed
  * target  = per-joint mean of the demos' first frames
  * envelope= the demos' own min/max, plus a margin (min/max, never quantiles:
    Mistake #8 fenced off the real grasp poses by using percentiles)
  * per-tick clamp enforced by the robot's own max_relative_target
  * stall detection, so it stops rather than fighting an obstruction
  * time cap
  * torque HELD on exit -- the entire point, since a limp arm sags
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SO101 = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", SO101 / "datasets" / "pick_place_v3"))
PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
CALIB = os.environ.get("FOLLOWER_CALIB", os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"))

STEP = float(os.environ.get("STEP_DEG", "4.0"))        # per tick; >3 beats gravity stiction
HZ = float(os.environ.get("GOTO_HZ", "5"))             # so ~20 deg/s
TOL = float(os.environ.get("TOL_DEG", "1.0"))          # arrival tolerance
TIME_CAP_S = float(os.environ.get("TIME_CAP_S", "45"))
MAX_TRAVEL = float(os.environ.get("MAX_TRAVEL_DEG", "70"))   # refuse absurd journeys
MAX_TRAVEL_GRIP = float(os.environ.get("MAX_TRAVEL_GRIP", "80"))
MARGIN = float(os.environ.get("MARGIN_DEG", "5"))      # envelope slack around demo min/max

DRY = "--dry" in sys.argv
YES = "--yes" in sys.argv or "-y" in sys.argv
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; B = "\033[1m"; X = "\033[0m"


def fail(msg: str) -> None:
    print(f"{R}{msg}{X}", file=sys.stderr)
    raise SystemExit(1)


# ------------------------------------------------------- target, from the data
if not (DATASET_ROOT / "meta" / "info.json").exists():
    fail(f"no dataset at {DATASET_ROOT}")
info = json.loads((DATASET_ROOT / "meta" / "info.json").read_text())
# The dataset already suffixes these with '.pos'; strip it so we hold bare motor
# names and re-add the suffix for the robot API, which works either way.
raw = info["features"]["observation.state"].get("names") or []
motors = [str(n).removesuffix(".pos") for n in raw]
if not motors:
    fail("dataset does not name the state dimensions — cannot map them to motors")

starts = []
for f in sorted(glob.glob(str(DATASET_ROOT / "data" / "**" / "*.parquet"), recursive=True)):
    t = pq.read_table(f).to_pandas()
    if "episode_index" not in t:
        continue
    for _ep, g in t.groupby("episode_index"):
        starts.append(np.asarray(g.sort_values("frame_index").iloc[0]["observation.state"], float))
if not starts:
    fail(f"no episodes found under {DATASET_ROOT}")

S = np.array(starts)
if S.shape[1] != len(motors):
    fail(f"state width {S.shape[1]} != {len(motors)} named motors — refusing to guess the mapping")
target = S.mean(axis=0)
lo, hi = S.min(axis=0) - MARGIN, S.max(axis=0) + MARGIN
is_grip = np.array(["gripper" in m for m in motors])

print(f"{B}GOTO START{X}  dataset={DATASET_ROOT.name}  ({len(S)} episodes)")

# ------------------------------------------- current pose, WITHOUT energising
# Same approach as read_pose.py: talk to the bus directly and only sync_read.
# SO101Follower.connect() would run configure(), which re-enables torque -- not
# something --dry should ever do behind the operator's back.
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

if not os.access(PORT, os.R_OK | os.W_OK):
    fail(f"cannot access {PORT} — not in the dialout group?\n"
         f"  try: sg dialout -c 'python {' '.join(sys.argv)}'")
if not os.path.exists(CALIB):
    fail(f"no follower calibration at {CALIB}")

MOTORS = {m: Motor(i + 1, "sts3215",
                   MotorNormMode.RANGE_0_100 if is_grip[i] else MotorNormMode.DEGREES)
          for i, m in enumerate(motors)}
calib = {n: MotorCalibration(**c) for n, c in json.load(open(CALIB)).items()}
bus = FeetechMotorsBus(port=PORT, motors=MOTORS, calibration=calib)
bus.connect()
try:
    acc = {m: [] for m in motors}
    for _ in range(20):
        for n, v in bus.sync_read("Present_Position", normalize=True, num_retry=3).items():
            acc[n].append(v)
        time.sleep(1 / 60)
finally:
    bus.disconnect(disable_torque=False)          # leave torque exactly as found
cur = np.array([float(np.mean(acc[m])) for m in motors])

# ------------------------------------------------------------------- the plan
delta = target - cur
print(f"\n  {'joint':<15}{'now':>9}{'target':>9}{'move':>9}   demo envelope")
for i, m in enumerate(motors):
    print(f"  {m:<15}{cur[i]:>9.1f}{target[i]:>9.1f}{delta[i]:>+9.1f}   [{lo[i]:.0f}, {hi[i]:.0f}]")

arm_travel = float(np.abs(delta[~is_grip]).max()) if (~is_grip).any() else 0.0
grip_travel = float(np.abs(delta[is_grip]).max()) if is_grip.any() else 0.0
print(f"\n  largest arm move: {arm_travel:.1f} deg   gripper: {grip_travel:.1f} units")

if arm_travel > MAX_TRAVEL or grip_travel > MAX_TRAVEL_GRIP:
    fail(f"refusing: {arm_travel:.1f} deg / {grip_travel:.1f} units exceeds the travel cap "
         f"({MAX_TRAVEL:.0f}/{MAX_TRAVEL_GRIP:.0f}). Wrong dataset, or the arm is somewhere "
         f"unexpected — reposition by hand (python so101/relax.py), then re-check.")
if np.any(cur < lo - MAX_TRAVEL) or np.any(cur > hi + MAX_TRAVEL):
    fail("refusing: the arm is far outside the demos' envelope — reposition by hand first")

if max(arm_travel, grip_travel) <= TOL:
    print(f"{G}{B}  already at the demo start pose — nothing to do.{X}")
    raise SystemExit(0)
if DRY:
    print(f"{Y}  --dry: planned only. Torque untouched, the arm did not move.{X}")
    raise SystemExit(0)
if not YES:
    print(f"\n  {B}The arm will MOVE{X} at ~{STEP * HZ:.0f} deg/s. Ctrl-C stops and holds.")
    if input("  Workspace clear? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")

# ------------------------------------------------------------------ the move
from lerobot.robots.so_follower import SO101Follower  # noqa: E402
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig  # noqa: E402

robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower", max_relative_target=STEP))
robot.connect()                                   # energises, holding where it stands
try:
    def read() -> np.ndarray:
        obs = robot.get_observation()
        return np.array([float(obs[f"{m}.pos"]) for m in motors])

    # Command the final target every tick and let max_relative_target walk it
    # there STEP at a time, so the clamp cannot be bypassed by this loop.
    goal = {f"{m}.pos": float(target[i]) for i, m in enumerate(motors)}
    t0 = time.perf_counter()
    prev = read()
    stalled = 0
    try:
        while True:
            robot.send_action(goal)
            time.sleep(1.0 / HZ)
            now = read()
            err = np.abs(target - now)
            if err.max() <= TOL:
                break
            stalled = stalled + 1 if np.abs(now - prev).max() < 0.15 else 0
            if stalled >= 6:
                print(f"\n{R}  STALLED{X} — {err.max():.1f} off target at "
                      f"{motors[int(np.argmax(err))]} and not moving. Obstruction, or the "
                      f"servo cannot hold this load. Stopped; torque held.")
                break
            if time.perf_counter() - t0 > TIME_CAP_S:
                print(f"\n{R}  TIME CAP{X} — {err.max():.1f} off target after {TIME_CAP_S:.0f}s. "
                      f"Stopped; torque held.")
                break
            prev = now
            print(f"\r  moving… worst error {err.max():5.1f}", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n{Y}  Ctrl-C — stopped here, torque held.{X}")

    final = read()
    err = np.abs(target - final)
    print(f"\n\n  {'joint':<15}{'final':>9}{'target':>9}{'error':>9}")
    for i, m in enumerate(motors):
        print(f"  {m:<15}{final[i]:>9.1f}{target[i]:>9.1f}{err[i]:>+9.1f}")
    ok = bool(err.max() <= TOL)
    print(f"\n{(G + B + '  AT START POSE') if ok else (Y + B + '  NOT CONVERGED')}{X}"
          f" — worst error {err.max():.1f} (tolerance {TOL:.1f})")
    print("  Torque is held, so it will not sag before the rollout.")
    raise SystemExit(0 if ok else 1)
finally:
    # Never leave the arm limp: sagging between trials is the whole problem.
    robot.config.disable_torque_on_disconnect = False
    robot.disconnect()
