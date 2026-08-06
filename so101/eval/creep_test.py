#!/usr/bin/env python
"""
SAFE creep-speed rollout test for the 150k ACT policy on the real SO101 follower.

Modes:
  python creep_test.py           DRY RUN (default) — NO MOTION. Connects (arm just
                                 holds its current pose), reads the live pose, runs
                                 ONE inference on the real camera+state, prints what
                                 it WOULD command, exits still holding pose.
  python creep_test.py --go      The creep test. Every guardrail below active.
  python creep_test.py --selftest  No robot, no camera, no motion: unit-tests the
                                 grip detector + retry targets, then runs one full
                                 inference on a synthetic observation. Safe anywhere.

Env knobs (baked into ~/creep — never type them as a command prefix, Mistake #12):
  NSTEP=50    re-plan interval in ticks (chunk is 100; 50 = one mid-descent correction)
  RETRY=1     auto-retry on a failed close: fires on close-on-air (grip < 20
              sustained 0.6 s) OR jam/peck/freeze (12 s below wide-open, calibrated
              on runs #16-20) => open jaws, re-home to demo start with alternating
              pan offsets (max MAX_RETRIES=3). Misses are correlated within a run
              (#14/#15), so the offset makes a retry a fresh draw, not a repeat.
  CAMS=front,wrist + DATASET_ROOT/DATASET_REPO/MODEL   v2 (wrist-cam) policy rollouts.

Guardrails (--go):
  * max_relative_target = a per-motor per-tick clamp (the CLAMP dict below; the
    exact values are printed into the run log) -> the arm physically cannot move
    faster than clamp*30 deg/s on any joint.
  * n_action_steps re-plans every NSTEP ticks (default 25 with FAST, else 10;
    ~/creep bakes NSTEP=100) instead of running a full 3.3 s plan open-loop.
  * Every commanded position clamped to the demos' FULL min..max action envelope —
    the policy cannot command a pose the demos never visited. (q01..q99 was tried
    and fenced the grasp off in the 1% tails; see the env_lo/env_hi note.)
  * Duration-capped (default 20 s). Ctrl-C stops instantly.
  * Bus-error watchdog: 5 consecutive failures -> abort.
  * disable_torque_on_disconnect=False -> the arm HOLDS POSE on exit, never drops.
    (Use relax.py later if you want it limp.)
"""
import os, sys, time
import faulthandler; faulthandler.enable()   # print a traceback into the log on segfault
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action
from lerobot.utils.feature_utils import hw_to_dataset_features

GO = "--go" in sys.argv
POSE = "--pose" in sys.argv
SELFTEST = "--selftest" in sys.argv
MODEL = os.environ.get("MODEL", "/home/anvil/SO101_policy/training/run150k/checkpoints/last/pretrained_model")
ROOT = os.environ.get("DATASET_ROOT", "/home/anvil/SO101_policy/datasets/pick_place")
REPO = os.environ.get("DATASET_REPO", "local/pick_place")
# The object is a ~7 cm BLUE PUCK — "cube" is the datasets' baked-in task string
# (recording-time label); this constant must keep matching the data verbatim.
TASK = "Pick up the cube and place it in the bin"
FPS = 30
DURATION_S = float(os.environ.get("DURATION", "20"))
# FAST=1: clamps x2 and longer plan-commitment. The demos' grasp is a fast committed
# motion; at creep pace + 10-tick re-plans the policy churns at the hover (runs 4/6).
FAST = os.environ.get("FAST", "0") == "1"
N_ACTION_STEPS = int(os.environ.get("NSTEP", "25" if FAST else "10"))
# RETRY=1: when a close ends on air (grip < 20 sustained), re-home to the demo
# start pose with a small alternating pan offset and let the policy try again.
RETRY = os.environ.get("RETRY", "0") == "1"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# CAMS=front (v1 policy) or CAMS=front,wrist (v2 policy trained with the wrist cam).
# Both roles come from the pin files written by eval/cam_check.py — no hardcoded
# devices: the Sonix that was the v1 front cam is the v2 WRIST cam since 07-21.
CAMS = [c.strip() for c in os.environ.get("CAMS", "front").split(",") if c.strip()]
CAM_PINS = {"front": ("/home/anvil/SO101_policy/front_cam.path", "--find-front"),
            "wrist": ("/home/anvil/SO101_policy/wrist_cam.path", "--find")}


def camera_configs():
    cams = {}
    for name in CAMS:
        if name not in CAM_PINS:
            sys.exit(f"unknown camera '{name}' in CAMS")
        pin_file, find_flag = CAM_PINS[name]
        if not os.path.exists(pin_file):
            if SELFTEST:                      # selftest never opens cameras — any path works
                cams[name] = OpenCVCameraConfig(index_or_path=f"/dev/unpinned_{name}",
                                                width=1280, height=720, fps=30, fourcc="MJPG")
                continue
            sys.exit(f"{name} camera not configured — run:  "
                     f"python /home/anvil/SO101_policy/eval/cam_check.py {find_flag}")
        p = open(pin_file).read().strip()
        if not SELFTEST and not os.path.exists(p):
            sys.exit(f"{name} camera path {p} is gone — replug it, or re-run cam_check.py {find_flag}")
        cams[name] = OpenCVCameraConfig(index_or_path=p, width=1280, height=720, fps=30, fourcc="MJPG")
    if len(set(os.path.realpath(c.index_or_path) for c in cams.values())) < len(cams):
        sys.exit("two camera roles are pinned to the SAME device — UVC cams are "
                 "single-reader, so this cannot run. Re-pin one (cam_check.py --set/--set-front).")
    return cams


class GripMonitor:
    """Close-on-air / hold detector from the MEASURED gripper position.

    Demo forensics (CHECKPOINT.md §3): demos carry the puck at grip 26-36; a
    close that settles below ~20 means the jaws passed through where the puck
    body should be (this called runs #13-15 correctly). The gripper starts
    CLOSED (~3.5) at the demo start pose, so the detector only arms after the
    policy has OPENED the jaws (> 38) — no false trigger at start or re-home.
    """
    OPEN_THR = 34.0   # run 2 (2026-07-20) opened to only 35.6 — 38 never armed; carries sit 26-31
    WIDE_THR = 37.5   # only a genuinely WIDE open resets the clocks — the peck
                      # loop re-opens to 34-36 (run 5), which must NOT reset jam
    AIR_THR = 20.0
    HOLD_LO, HOLD_HI = 24.0, 38.0
    AIR_SUSTAIN_S = 0.6
    HOLD_SUSTAIN_S = 1.0
    # armed + below-wide for this long = jam/peck/freeze, not a task: #13's
    # success spends ≤~11 s below-wide (5 s narrow-open descend + 6 s carry);
    # tonight's failures ground below-wide for 20-70 s
    JAM_SUSTAIN_S = 12.0

    def __init__(self, fps):
        self.fps = fps
        self.reset()

    def reset(self):
        self.armed = False
        self._below = 0
        self._inband = 0
        self._subwide = 0
        self._hold_announced = False

    def update(self, grip):
        """Feed one measured gripper position. Returns 'air', 'jam', 'hold', or None."""
        if grip > self.OPEN_THR:
            self.armed = True
        if grip >= self.WIDE_THR:             # genuinely wide open: clocks reset
            self._below = 0
            self._inband = 0
            self._subwide = 0
            self._hold_announced = False
            return None
        if not self.armed:
            return None
        self._below = self._below + 1 if grip < self.AIR_THR else 0
        self._inband = self._inband + 1 if self.HOLD_LO <= grip <= self.HOLD_HI else 0
        self._subwide += 1                    # any time below wide-open counts
        if self._below >= int(self.AIR_SUSTAIN_S * self.fps):
            self.reset()                      # disarm until the next open
            return "air"
        if self._subwide >= int(self.JAM_SUSTAIN_S * self.fps):
            self.reset()                      # disarm until the next open
            return "jam"
        if not self._hold_announced and self._inband >= int(self.HOLD_SUSTAIN_S * self.fps):
            self._hold_announced = True
            return "hold"
        return None


# Retry pan offsets (deg). ~2.5 deg of shoulder_pan is ~1.2 cm laterally at the
# puck (~28 cm reach). The offset is the point: misses are correlated WITHIN a
# run (#14 all right-rear, #15 all left-near) — an unperturbed retry re-executes
# the same miss, so alternate sides, then go wider.
RETRY_PAN_OFFSETS = [2.5, -2.5, 5.0]

print(f"mode: {'SELFTEST (no robot, no motion)' if SELFTEST else 'POSE (limp + live readout)' if POSE else 'GO (creep, guard-railed)' if GO else 'DRY RUN (no motion)'}")

# ---- POSE mode: relax the arm, live joint readout while you hand-pose it ----
if POSE:
    import select
    # v2 demo-mean start — mean of ALL 45 first frames of the final session
    # (measured 07-21 23:15). Note the demos' own starts spread sigma=7.6deg in
    # pan (operator variance) — the policy saw variety, so matching the mean
    # loosely is fine; TOL below already covers it.
    TARGETS = [-40.1, -100.6, 96.5, 72.4, -4.8, 1.4]
    TOL     = [15, 15, 15, 15, 15, 10]
    robot = SO101Follower(SO101FollowerConfig(port="/dev/so101_follower", id="follower",
                          cameras={}, disable_torque_on_disconnect=False))
    print("connecting ...")
    robot.connect()
    print("\n*** SUPPORT THE ARM — going LIMP in 3 seconds ***")
    time.sleep(3)
    robot.bus.disable_torque()
    try:
        while True:
            obs = robot.get_observation()
            rows = ["\033[H\033[2J  LIVE POSE — move the arm by hand", ""]
            rows.append(f"  {'joint':<14} {'live':>7} {'target':>7} {'delta':>7}   status")
            ok_all = True
            for j, n in enumerate(JOINTS):
                v = obs[f"{n}.pos"]; d = v - TARGETS[j]
                ok = abs(d) <= TOL[j]; ok_all = ok_all and ok
                mark = "\033[32mOK\033[0m" if ok else f"\033[33m<-- move {'-' if d>0 else '+'}{abs(d):.0f}\033[0m"
                rows.append(f"  {n:<14} {v:>7.1f} {TARGETS[j]:>7.1f} {d:>+7.1f}   {mark}")
            rows.append("")
            rows.append("  \033[32m*** ALL SET — press ENTER to LOCK ***\033[0m" if ok_all
                        else "  adjust the yellow joints ...")
            rows.append("  ENTER = lock this pose and hold | Ctrl-C = leave limp and exit")
            print("\n".join(rows), flush=True)
            r, _, _ = select.select([sys.stdin], [], [], 0.15)
            if r:
                sys.stdin.readline()
                break
    except KeyboardInterrupt:
        print("\nLeft LIMP (torque off). Arm is free.")
        robot.disconnect()
        sys.exit(0)
    robot.bus.enable_torque()
    print("\nLOCKED — torque on, holding this pose. Now run:  ~/creep go")
    robot.disconnect()
    sys.exit(0)

# ---- dataset meta (features + stats; no video machinery) -------------------
meta = LeRobotDatasetMetadata(REPO, root=ROOT)
st = meta.stats["action"]
# FULL demo action range (min..max), not q01..q99 — the grasp/place moments are the
# rarest frames, so their poses live in the 1% tails; clipping them fenced the policy
# 5-8 deg above the puck for 80 s (run #4). min..max is still 100% demo-bounded.
env_lo = np.array(st["min"], dtype=np.float32)
env_hi = np.array(st["max"], dtype=np.float32)
demo_start = np.array([-40.1, -100.6, 96.5, 72.4, -4.8, 1.4])  # v2 demo-mean (final: all 45 demos, measured 07-21 23:15)


def retry_target(k):
    """Demo start pose + the k-th pan offset, clipped to the demo envelope."""
    tgt = demo_start.copy()
    tgt[0] += RETRY_PAN_OFFSETS[min(k, len(RETRY_PAN_OFFSETS) - 1)]
    return np.clip(tgt, env_lo, env_hi).astype(np.float32)

# ---- policy ----------------------------------------------------------------
cfg = PreTrainedConfig.from_pretrained(MODEL)
cfg.device = "cpu"
policy = ACTPolicy.from_pretrained(MODEL); policy.to("cpu")
policy.eval()
# Mistake-#1 guard, EVERY run: prove the in-memory params are this checkpoint's
# bytes. "Loading weights..." printing is not proof — an unset pretrained_path
# gives a silent random net with the identical log line. Costs ~1-2 s.
from safetensors import safe_open as _safe_open
_sd = policy.state_dict()
_nbad = _nok = 0
with _safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu") as _f:
    for _k in _f.keys():
        _t = _f.get_tensor(_k)
        _m = _sd.get(_k)
        if _m is None or _m.shape != _t.shape or not torch.equal(_m, _t):
            _nbad += 1
        else:
            _nok += 1
if _nbad:
    sys.exit(f"WEIGHTS MISMATCH: {_nbad}/{_nbad + _nok} tensors differ from "
             f"{MODEL}/model.safetensors — random-net risk (Mistake #1). ABORTING.")
print(f"weights verified: {_nok}/{_nok} tensors byte-identical to model.safetensors "
      f"({os.path.realpath(MODEL)})")
policy.config.n_action_steps = N_ACTION_STEPS
_dev = {"device_processor": {"device": "cpu"}}
pre, post = make_pre_post_processors(cfg, pretrained_path=MODEL, dataset_stats=meta.stats,
                                     preprocessor_overrides=_dev, postprocessor_overrides=_dev)

# ---- robot -----------------------------------------------------------------
# per-tick physical clamp, per-motor: gravity-loaded joints need ~3 deg of
# position error to generate enough torque to actually lift (1 deg stalls). This
# is the REAL creep speed limit; it is printed into the run log below (so the log
# never lies about the clamp) and is None off --go so nothing moves.
CLAMP = ({
    "shoulder_pan": 4.0, "shoulder_lift": 6.0, "elbow_flex": 6.0,
    "wrist_flex": 4.0, "wrist_roll": 4.0, "gripper": 10.0,
} if FAST else {
    "shoulder_pan": 2.0, "shoulder_lift": 3.0, "elbow_flex": 3.0,
    "wrist_flex": 2.0, "wrist_roll": 2.0, "gripper": 5.0,
}) if GO else None
robot_cfg = SO101FollowerConfig(
    port="/dev/so101_follower",
    id="follower",
    cameras=camera_configs(),
    max_relative_target=CLAMP,
    disable_torque_on_disconnect=False,                 # NEVER drop the arm on exit
)
robot = SO101Follower(robot_cfg)
obs_feats = hw_to_dataset_features(robot.observation_features, "observation")
act_feats = hw_to_dataset_features(robot.action_features, "action")

# build_dataset_frame equivalent (state + camera keys)
from lerobot.utils.feature_utils import build_dataset_frame

# GO runs self-record the policy camera (~6 fps) so every rollout can be reviewed
RECDIR = None
if GO:
    RECDIR = os.path.join("/home/anvil/SO101_policy/eval/run_recordings", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(RECDIR, exist_ok=True)
    print(f"(recording robot-view frames to {RECDIR})")
_fi = [0]

def snap(obs):
    """Save every 3rd frame of EVERY policy camera into the run recording
    (~6 fps footage). f*.jpg = front, w*.jpg = wrist. Wrist added 07-22 after
    run #21 — the grasp-phase wrist view is the primary evidence for the
    localization question, and run #21's is lost because only front was saved."""
    if RECDIR is None:
        return
    _fi[0] += 1
    if _fi[0] % 3 == 1:
        import cv2 as _cv
        _cv.imwrite(f"{RECDIR}/f{_fi[0]:05d}.jpg",
                    _cv.cvtColor(obs["front"], _cv.COLOR_RGB2BGR),
                    [_cv.IMWRITE_JPEG_QUALITY, 80])
        if "wrist" in obs:
            _cv.imwrite(f"{RECDIR}/w{_fi[0]:05d}.jpg",
                        _cv.cvtColor(obs["wrist"], _cv.COLOR_RGB2BGR),
                        [_cv.IMWRITE_JPEG_QUALITY, 80])

def infer_once():
    obs = robot.get_observation()
    snap(obs)
    frame = build_dataset_frame(obs_feats, obs, prefix="observation")
    observation = prepare_observation_for_inference(frame, "cpu", TASK, robot.name)
    with torch.inference_mode():
        a = post(policy.select_action(pre(observation)))
    a = a.squeeze(0).float().cpu().numpy()
    a = np.clip(a, env_lo, env_hi)                       # demo-envelope clamp
    state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    return state, a


def do_retry(k):
    """Guard-railed re-home to the demo start pose with a small pan offset, then
    hand control back to a freshly-reset policy (clears the stale action chunk).
    Moves through send_action, so the per-tick clamp shapes the whole motion."""
    tgt = retry_target(k)
    print(f"\n  ↺ RETRY {k + 1}/{MAX_RETRIES}: re-homing to demo start "
          f"(pan offset {tgt[0] - demo_start[0]:+.1f} deg) ...")
    # 1) open the jaws in place first — a jammed close pinches the puck's rim;
    #    re-homing while pinched would drag or flick the puck across the table
    t_open = time.time() + 0.6
    while time.time() < t_open:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        snap(obs)
        state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        rel = state.copy()
        rel[5] = min(40.0, float(env_hi[5]))
        robot.send_action(make_robot_action(torch.from_numpy(rel), act_feats))
        dt = time.perf_counter() - t0
        if dt < 1 / FPS:
            time.sleep(1 / FPS - dt)
    # 2) guard-railed re-home
    deadline = time.time() + 5.0
    state = None
    while time.time() < deadline:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        snap(obs)                              # keep the run footage continuous
        state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        if np.abs(state[:5] - tgt[:5]).max() < 3.0 and abs(state[5] - tgt[5]) < 8.0:
            break
        robot.send_action(make_robot_action(torch.from_numpy(tgt), act_feats))
        dt = time.perf_counter() - t0
        if dt < 1 / FPS:
            time.sleep(1 / FPS - dt)
    if state is not None:
        print(f"  ↺ re-homed (max arm delta {np.abs(state[:5] - tgt[:5]).max():.1f} deg) — policy resumes fresh.")
    policy.reset(); pre.reset(); post.reset()


# ---- SELFTEST: everything except serial/camera I/O — safe anywhere ---------
if SELFTEST:
    print("\n=== SELFTEST (no robot, no camera, no motion) ===")
    fails = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            fails.append(name)

    # 1) GripMonitor state machine on synthetic traces (30 Hz ticks)
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 150]
    check("start pose (grip 3.5) never triggers", not any(evs))
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 30 + [41.0] * 60 + [30.0] * 45]
    check("run#13-style open->close-to-30 => HOLD, no AIR", evs.count("hold") == 1 and "air" not in evs)
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 30 + [41.0] * 60 + [8.0] * 30]
    check("run#14-style open->close-to-8 => AIR", "air" in evs and "hold" not in evs)
    evs2 = [m.update(g) for g in [3.5] * 30 + [41.0] * 30 + [31.0] * 45]
    check("re-arms after AIR; next close-to-31 => HOLD", evs2.count("hold") == 1 and "air" not in evs2)
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 15 + [41.0] * 30 + [30.0] * 15 + [8.0] * 10 + [41.0] * 20]
    check("brief transients (<0.6 s) never trigger", not any(evs))
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 15 + [41.0] * 30 + [28.0] * 380]
    check("rim-jam (below-wide 12.7 s) => JAM", "jam" in evs and "air" not in evs)
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 15 + [36.0] * 150 + ([27.0] * 60 + [35.0] * 30) * 4]
    check("run#5-style peck loop (re-opens to 35) => JAM", "jam" in evs and "air" not in evs)
    m = GripMonitor(FPS)
    evs = [m.update(g) for g in [3.5] * 15 + [41.0] * 30 + [30.0] * 180 + [41.0] * 15]
    check("run#13-style 6 s carry never JAMs", "jam" not in evs and evs.count("hold") == 1)

    # 2) retry targets: alternating pan offsets, always demo-bounded
    ts = [retry_target(k) for k in range(MAX_RETRIES)]
    check("retry pan offsets alternate sides", ts[0][0] > demo_start[0] and ts[1][0] < demo_start[0])
    check("retry targets inside demo envelope", all(((t >= env_lo).all() and (t <= env_hi).all()) for t in ts))

    # 3) the full inference path on a synthetic observation — exactly the GO
    #    code path (build frame -> pre -> policy -> post -> envelope clip),
    #    minus robot/camera I/O
    fake = {f"{j}.pos": float(demo_start[i]) for i, j in enumerate(JOINTS)}
    for c in CAMS:
        fake[c] = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = build_dataset_frame(obs_feats, fake, prefix="observation")
    observation = prepare_observation_for_inference(frame, "cpu", TASK, robot.name)
    t0 = time.perf_counter()
    with torch.inference_mode():
        a = post(policy.select_action(pre(observation)))
    a = np.clip(a.squeeze(0).float().cpu().numpy(), env_lo, env_hi)
    dt = time.perf_counter() - t0
    check("inference returns 6 finite joint targets", a.shape == (6,) and bool(np.isfinite(a).all()))
    check(f"n_action_steps == {N_ACTION_STEPS}", policy.config.n_action_steps == N_ACTION_STEPS)
    print(f"  (one inference: {dt:.2f}s on CPU · model {MODEL})")
    print(f"  (cams: {CAMS} · retry {'ON max ' + str(MAX_RETRIES) if RETRY else 'off'} · envelope + clamp active)")
    print(f"\nSELFTEST: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(0 if not fails else 1)

print("connecting (arm will hold its current pose) ...")
robot.connect()
try:
    policy.reset(); pre.reset(); post.reset()
    state, target = infer_once()
    print("\n=== GROUNDING CHECK ===")
    print(f"{'joint':<14} {'LIVE now':>9} {'demo start':>10} {'policy tgt':>10} {'delta':>7}")
    for j, name in enumerate(JOINTS):
        print(f"{name:<14} {state[j]:>9.1f} {demo_start[j]:>10.1f} {target[j]:>10.1f} {target[j]-state[j]:>7.1f}")
    far = np.abs(state[:4] - demo_start[:4]).max()
    print(f"\nmax |live - demo start| (arm joints): {far:.1f} deg "
          f"{'(OK, near demo start)' if far < 20 else '(FAR from demo start — reposition first!)'}")
    big = np.abs(target - state).max()
    print(f"max |first commanded move|: {big:.1f} deg")

    if not GO:
        print("\nDRY RUN done — no motion commanded. Arm still holding pose.")
    elif far > 25:
        print("\nABORT: arm is >25 deg from the demo start pose. Not moving. Re-pose and retry.")
    else:
        print(f"\nCREEP TEST: {DURATION_S:.0f}s, per-tick clamp {CLAMP}, re-plan every {N_ACTION_STEPS} ticks, "
              f"retry={'ON (max ' + str(MAX_RETRIES) + ', pan offsets ' + str(RETRY_PAN_OFFSETS) + ')' if RETRY else 'off'}.")
        print("Ctrl-C stops instantly. Starting in 3s — hands clear, watch the arm.")
        time.sleep(3)
        t_end = time.time() + DURATION_S
        tick = 0; bus_errs = 0; retries = 0
        monitor = GripMonitor(FPS)
        policy.reset(); pre.reset(); post.reset()
        while time.time() < t_end:
            t0 = time.perf_counter()
            try:
                state, a = infer_once()
                action = make_robot_action(torch.from_numpy(a), act_feats)
                robot.send_action(action)
                bus_errs = 0
                # grip verdicts (always printed; acted on only with RETRY=1)
                ev = monitor.update(state[5])
                if ev == "hold":
                    print(f"  ● t={DURATION_S-(t_end-time.time()):5.1f}s grip settled at {state[5]:.1f} "
                          f"— carry band (26-36): HOLD *or* rim-jam (grip can't tell; "
                          f"real hold = arm lifts with the puck — check the video)")
                elif ev in ("air", "jam"):
                    what = ("close-on-air (grip sustained <20)" if ev == "air"
                            else "jam/peck/freeze (12s below wide-open without progress)")
                    print(f"  ✗ t={DURATION_S-(t_end-time.time()):5.1f}s {what}, grip {state[5]:.1f}"
                          + (" — retry budget exhausted" if RETRY and retries >= MAX_RETRIES else ""))
                    if RETRY and retries < MAX_RETRIES:
                        do_retry(retries)
                        retries += 1
                        monitor.reset()
                        tick += 1
                        continue
                # Log INSIDE the try, after a successful infer: `a`/`state` are
                # only defined on a good tick, so logging outside would NameError
                # on `a` if the very first tick errored — crashing the rollout and
                # defeating the 5-error watchdog on a single transient glitch.
                if tick % N_ACTION_STEPS == 0:
                    d = np.abs(a - state).max()
                    print(f"  t={DURATION_S-(t_end-time.time()):5.1f}s  max|tgt-now|={d:5.1f}  "
                          f"lift={state[1]:6.1f} elbow={state[2]:6.1f} grip={state[5]:5.1f}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                bus_errs += 1
                print(f"  bus/infer error ({bus_errs}/5): {type(e).__name__}: {e}")
                if bus_errs >= 5:
                    print("ABORT: too many consecutive errors."); break
            tick += 1
            dt = time.perf_counter() - t0
            if dt < 1/FPS:
                time.sleep(1/FPS - dt)
        print("\nCreep test finished.")
except KeyboardInterrupt:
    print("\nCtrl-C — stopping.")
finally:
    print("disconnecting (torque stays ON, arm holds pose) ...")
    robot.disconnect()
    print("done.")
    # Post-run scoring: label the recording OK/NG (renames the dir, encodes
    # labeled videos). tcflush first — buffered ENTERs auto-answer prompts
    # (the guided_record.py bug, 07-21).
    if GO and RECDIR is not None:
        _score = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_run.py")
        try:
            if sys.stdin.isatty():
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
                ans = input("\nscore this run — o=OK  n=NG  ENTER=skip: ").strip().lower()
                if ans in ("o", "n"):
                    import subprocess
                    subprocess.run([sys.executable, _score, "OK" if ans == "o" else "NG"])
                else:
                    print(f"unscored — label later with: python {_score} OK|NG")
            else:
                print(f"\n(recording: {RECDIR} — label it: python {_score} OK|NG)")
        except Exception as e:
            print("scoring prompt skipped:", e)
