#!/usr/bin/env python3
"""Re-zero the LEADER's homing offsets so its CURRENT physical pose reads
identically to the FOLLOWER's current pose — without moving either arm.

    python eval/match_leader.py

Why: teleop connect snaps the follower to the leader's *reading*. After the
07-21 base rotation + wrist-camera mount, the two arms sat in matching task
poses but the leader's readings still reflected v1 (pan −52°, roll +170° off)
— a connect would have whipped the wrist and torn the camera cable. Rather
than hand-moving the leader, this redefines its calibration so
current-physical == follower-reading. The FOLLOWER is never written.

Method (no guessed constants — everything measured or read back):
  1. Read follower normalized pose (read-only) -> per-joint targets.
  2. Leader: assert torque OFF on every motor; assert the motor-register
     homing values match leader.json (no silent drift).
  3. Empirically measure the homing->reading slope with a ±20-tick write that
     is immediately reverted (expected −1 per the Feetech doc
     Present = Actual − Homing, but MEASURED, not assumed).
  4. Per joint: H_new = H_old + (R_target − R_now)/slope, folded by ±4096 into
     the encodable sign-magnitude range [−2047, 2047] (fold validity is proven
     by the read-back check, not assumed).
  5. Write, read the register back, re-read the normalized position; if the
     reading is off by > 0.5° the original offset is restored and we abort.
  6. On success: update the calibration JSON (cache + repo copy), print the
     final leader-vs-follower table. Then run eval/snap_check.py as the
     independent gate.

DEGREES joints only. The gripper (RANGE_0_100) is left untouched — its delta
was 0.1 on a 0-100 scale.
"""
import json
import os
import shutil
import statistics
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CAL_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration")
LEADER_JSON = f"{CAL_DIR}/teleoperators/so_leader/leader.json"
FOLLOWER_JSON = f"{CAL_DIR}/robots/so_follower/follower.json"
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT as _ROOT, CALIB_DIR as _CALIB  # noqa: E402
REPO_LEADER_JSON = str(_ROOT / "leader.json")
BACKUP_DIR = str(_CALIB)
MAX_RES = 4095          # sts3215: MODEL_RESOLUTION 4096 − 1 (motors_bus.py:872)
ENC_LIMIT = 2047        # Homing_Offset sign-magnitude bit 11 (tables.py; verified)
TOL_DEG = 0.5           # post-write reading must match target within this
SKIP_TICKS = 3          # don't touch joints already this close
PROBE = 20              # slope-probe tick delta (reverted immediately)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def motor_table():
    return {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def load_cal(path):
    return {n: MotorCalibration(**c) for n, c in json.load(open(path)).items()}


def mean_read(bus, n=20, normalize=True):
    acc = {}
    for _ in range(n):
        for k, v in bus.sync_read("Present_Position", normalize=normalize, num_retry=3).items():
            acc.setdefault(k, []).append(v)
        time.sleep(1 / 60)
    return {k: statistics.fmean(v) for k, v in acc.items()}


def fold(h):
    """Fold a homing value by ±4096 into the encodable range, or None."""
    for cand in (h, h - 4096, h + 4096):
        if abs(cand) <= ENC_LIMIT:
            return cand
    return None


# ---- 1. follower targets (read-only, torque untouched) ----------------------
fbus = FeetechMotorsBus(port="/dev/so101_follower", motors=motor_table(),
                        calibration=load_cal(FOLLOWER_JSON))
fbus.connect()
target = mean_read(fbus, 30)
fbus.disconnect(disable_torque=False)
print("follower targets:", {k: round(v, 1) for k, v in target.items()})

# ---- 2. leader state + preconditions ---------------------------------------
lcal = load_cal(LEADER_JSON)
lbus = FeetechMotorsBus(port="/dev/so101_leader", motors=motor_table(), calibration=lcal)
lbus.connect()

for m in motor_table():
    tq = lbus.read("Torque_Enable", m, normalize=False)
    if tq != 0:
        lbus.disconnect(disable_torque=False)
        sys.exit(f"ABORT: leader {m} has torque ENABLED ({tq}) — expected a passive arm.")
print("precondition: leader torque OFF on all motors ✓")

reg_h = {m: int(lbus.read("Homing_Offset", m, normalize=False)) for m in motor_table()}
for m in motor_table():
    if reg_h[m] != lcal[m].homing_offset:
        lbus.disconnect(disable_torque=False)
        sys.exit(f"ABORT: leader {m} homing register {reg_h[m]} != leader.json "
                 f"{lcal[m].homing_offset} — calibration drift; resolve before matching.")
print("precondition: motor homing registers match leader.json ✓")

raw = mean_read(lbus, 20, normalize=False)
deg = mean_read(lbus, 20, normalize=True)

# ---- 3. backup, unlock, measure the slope empirically ----------------------
os.makedirs(BACKUP_DIR, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
backup = f"{BACKUP_DIR}/leader.pre-match-{stamp}.json"
shutil.copy2(LEADER_JSON, backup)
print(f"backup: {backup}")

lbus.disable_torque()   # torque already off; this also sets Lock=0 (EPROM unlock),
                        # the same pre-write state lerobot's own calibrate uses

probe_m = "shoulder_pan"
h0 = reg_h[probe_m]
r0 = raw[probe_m]
lbus.write("Homing_Offset", probe_m, h0 + PROBE)
time.sleep(0.05)
r1 = statistics.fmean(lbus.read("Present_Position", probe_m, normalize=False)
                      for _ in range(5))
lbus.write("Homing_Offset", probe_m, h0)          # revert immediately
time.sleep(0.05)
r_back = lbus.read("Present_Position", probe_m, normalize=False)
if abs(r_back - r0) > SKIP_TICKS:
    lbus.disconnect(disable_torque=False)
    sys.exit(f"ABORT: probe revert failed on {probe_m}: raw {r0:.0f} -> {r_back} — "
             f"restore homing {h0} by hand and investigate.")
delta_r = r1 - r0
if abs(abs(delta_r) - PROBE) > SKIP_TICKS:
    lbus.disconnect(disable_torque=False)
    sys.exit(f"ABORT: slope probe inconclusive (ΔH=+{PROBE} gave ΔR={delta_r:+.1f}); not writing.")
slope = 1 if delta_r > 0 else -1
print(f"slope probe on {probe_m}: ΔH=+{PROBE} -> ΔR={delta_r:+.1f}  => slope {slope:+d} ✓")

# ---- 4/5. per-joint: compute, write, verify (revert on any miss) -----------
changed = {}
print(f"\n  {'joint':<14} {'now':>7} {'target':>7} {'H_old':>6} -> {'H_new':>5}   result")
for m in JOINTS:
    mid = (lcal[m].range_min + lcal[m].range_max) / 2
    r_target = mid + target[m] * MAX_RES / 360.0
    d_r = r_target - raw[m]
    if abs(d_r) <= SKIP_TICKS:
        print(f"  {m:<14} {deg[m]:>7.1f} {target[m]:>7.1f} {reg_h[m]:>6}    (skip — already matched)")
        continue
    h_new = fold(round(reg_h[m] + d_r / slope))
    if h_new is None:
        print(f"  {m:<14} UNENCODABLE even after ±4096 fold — aborting")
        break
    lbus.write("Homing_Offset", m, h_new)
    time.sleep(0.05)
    stored = int(lbus.read("Homing_Offset", m, normalize=False))
    got = statistics.fmean(lbus.read("Present_Position", m, normalize=True) for _ in range(8))
    if stored != h_new or abs(got - target[m]) > TOL_DEG:
        lbus.write("Homing_Offset", m, reg_h[m])   # restore the original
        time.sleep(0.05)
        restored = statistics.fmean(lbus.read("Present_Position", m, normalize=True)
                                    for _ in range(8))
        print(f"  {m:<14} {deg[m]:>7.1f} {target[m]:>7.1f} {reg_h[m]:>6} -> {h_new:>5}   "
              f"FAIL (stored={stored}, read {got:.1f}) — reverted, now reads {restored:.1f}")
        break
    changed[m] = h_new
    print(f"  {m:<14} {deg[m]:>7.1f} {target[m]:>7.1f} {reg_h[m]:>6} -> {h_new:>5}   "
          f"✓ reads {got:.1f}")
else:
    # ---- 6. all writes verified: persist JSONs (cache + repo copy) ---------
    data = json.load(open(LEADER_JSON))
    for m, h in changed.items():
        data[m]["homing_offset"] = h
    for path in (LEADER_JSON, REPO_LEADER_JSON):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    print(f"\nsaved: {LEADER_JSON}\nsaved: {REPO_LEADER_JSON} (repo copy)")
    print("NOTE: ~/ALAN keeps its own copy of leader.json — sync it before any ALAN teleop.")

    final = mean_read(lbus, 20)
    lbus.disconnect(disable_torque=False)
    print(f"\n  FINAL  {'joint':<14} {'leader':>8} {'follower':>9} {'delta':>7}")
    for m in motor_table():
        print(f"         {m:<14} {final[m]:>8.1f} {target[m]:>9.1f} {final[m]-target[m]:>+7.1f}")
    print("\nDone. Now run the independent gate:  python eval/snap_check.py")
    sys.exit(0)

# a break above means something failed and was reverted
lbus.disconnect(disable_torque=False)
sys.exit("\nABORT: no JSON was modified; any written register was restored. "
         "Motor state matches the backup — investigate before retrying.")
