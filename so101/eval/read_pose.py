#!/usr/bin/env python3
"""Read the follower's current joint angles. READ-ONLY — never writes, never
touches torque, so it is safe to run while the arm is limp AND while it holds.

    python eval/read_pose.py [secs]

Prints the mean of `secs` (default 1.0) of 60 Hz samples, in the SAME units as
`creep_test.py`'s TARGETS/demo_start (arm joints in DEGREES, gripper 0-100) —
so the output can be pasted straight into those constants.

Why not robot.connect(): SOFollower.connect() calls configure(), whose
`bus.torque_disabled()` context manager GUARANTEES torque is re-enabled on
exit. Connecting to "just read" would silently stiffen a limp arm. This talks
to the bus directly (the probe_follower_read.py pattern) and only sync_reads.
"""
import json
import os
import statistics
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT as _ROOT  # noqa: E402
BASE = str(_ROOT)
PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
CALIB = os.environ.get(
    "FOLLOWER_CALIB", f"{os.path.expanduser('~')}/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
)
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

# Same motor table + norm modes the SO101Follower uses, so the numbers match.
MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
V1_START = [-40.2, -100.9, 95.3, 73.8, 165.0, 3.5]  # mean of the 30 v1 demo starts

if not os.path.exists(CALIB):
    sys.exit(f"no calibration at {CALIB}")
calibration = {n: MotorCalibration(**c) for n, c in json.load(open(CALIB)).items()}

bus = FeetechMotorsBus(port=PORT, motors=MOTORS, calibration=calibration)
bus.connect()
print(f"connected READ-ONLY to {PORT} (torque untouched)\n")

samples = {n: [] for n in MOTORS}
period = 1.0 / 60.0
t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < SECS:
        t = time.perf_counter()
        pos = bus.sync_read("Present_Position", normalize=True, num_retry=3)
        for n, v in pos.items():
            samples[n].append(v)
        dt = time.perf_counter() - t
        if dt < period:
            time.sleep(period - dt)
finally:
    bus.disconnect(disable_torque=False)  # leave torque exactly as we found it

n_s = len(samples["shoulder_pan"])
print(f"  {'joint':<14} {'live':>8} {'jitter':>8} {'v1 start':>9} {'delta':>8}")
means = []
for i, name in enumerate(MOTORS):
    vals = samples[name]
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    means.append(m)
    print(f"  {name:<14} {m:>8.1f} {sd:>8.2f} {V1_START[i]:>9.1f} {m - V1_START[i]:>+8.1f}")

print(f"\n  ({n_s} samples over {time.perf_counter() - t0:.1f}s)")
print("\n  paste-ready:")
print("  [" + ", ".join(f"{m:.1f}" for m in means) + "]")
