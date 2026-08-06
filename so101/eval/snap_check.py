#!/usr/bin/env python3
"""SNAP CHECK — read BOTH arms (read-only, torque untouched) and compare poses.

    python eval/snap_check.py

Run this BEFORE anything that connects leader+follower for teleop (~/rec,
teleop.sh): on connect the follower snaps to the leader's pose at full servo
speed, uncclamped. If the arms aren't matched, that snap can whip the wrist —
which, since 07-21, carries the wrist camera and its cable. A 170-degree
wrist_roll mismatch would tear it clean off. Added 07-21 at the user's callout.

Exit code 0 = safe to connect, 1 = mismatch (hand-adjust the LEADER to match
the follower — never the other way: the follower is at the recorded home pose).
"""
import json
import os
import statistics
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CAL = os.path.expanduser("~/.cache/huggingface/lerobot/calibration")
ARMS = {
    "leader": ("/dev/so101_leader", f"{CAL}/teleoperators/so_leader/leader.json"),
    "follower": ("/dev/so101_follower", f"{CAL}/robots/so_follower/follower.json"),
}
TOL_DEG = 5.0       # arm joints: max acceptable snap
TOL_GRIP = 8.0      # gripper (0-100 scale): sloppier is fine — no cable to tear


def motors():
    return {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


pose = {}
for name, (port, calib) in ARMS.items():
    cal = {n: MotorCalibration(**c) for n, c in json.load(open(calib)).items()}
    bus = FeetechMotorsBus(port=port, motors=motors(), calibration=cal)
    bus.connect()
    samples = {n: [] for n in motors()}
    for _ in range(30):
        for n, v in bus.sync_read("Present_Position", normalize=True, num_retry=3).items():
            samples[n].append(v)
        time.sleep(1 / 60)
    bus.disconnect(disable_torque=False)  # leave torque exactly as we found it
    pose[name] = {n: statistics.fmean(v) for n, v in samples.items()}

print("SNAP CHECK — on connect the follower jumps BY the delta below:")
print(f"  {'joint':<14} {'leader':>8} {'follower':>9} {'delta':>8}")
ok = True
for n in motors():
    d = pose["leader"][n] - pose["follower"][n]
    tol = TOL_GRIP if n == "gripper" else TOL_DEG
    good = abs(d) <= tol
    ok = ok and good
    mark = "OK" if good else "<<< MISMATCH — hand-move the LEADER to match"
    print(f"  {n:<14} {pose['leader'][n]:>8.1f} {pose['follower'][n]:>9.1f} {d:>+8.1f}  {mark}")

print()
if ok:
    print("VERDICT: SAFE TO CONNECT — the snap is negligible.")
else:
    print("VERDICT: DO NOT CONNECT — match the LEADER to the follower and re-run.")
sys.exit(0 if ok else 1)
