#!/usr/bin/env python
"""
Guided SO101 pick-and-place recorder — a clear, step-by-step wrapper around
lerobot's record_loop. No confusing async arrow keys: every phase is an
explicit ENTER-bounded step, and an empty episode can NEVER be saved.

Model:
  - During the RECORD phase the follower mirrors your leader arm (teleop) and
    frames are captured. When recording starts, the follower catches up to the
    leader's current pose — so position the leader near the arm before you start.
  - SET UP / RESET are plain ENTER prompts (no motion captured); the arm holds.
  - You press ENTER to move between clearly-labelled steps. That's the only key.

Env (set by so101/rec): TASK, EPISODES, DATASET_ROOT, WRIST, camera pin files.
"""
from __future__ import annotations

import os
import sys
import termios
import time
import threading
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop

from _paths import ROOT, DEFAULT_DATASET, FRONT_CAM_FILE, WRIST_CAM_FILE

# ---- config ----------------------------------------------------------------
TASK = os.environ.get("TASK", "pick the blue puck and place it in the brown box")
TARGET_ENV = os.environ.get("EPISODES", "").strip()
ROOT_DS = Path(os.environ.get("DATASET_ROOT", str(DEFAULT_DATASET))).expanduser()
REPO_ID = "local/" + ROOT_DS.name
FPS = 30
FRONT_CAM = Path(os.environ.get("FRONT_CAM_FILE", str(FRONT_CAM_FILE))).expanduser()
WRIST = os.environ.get("WRIST", "1") == "1"
WRIST_CAM = Path(os.environ.get("WRIST_CAM_FILE", str(WRIST_CAM_FILE))).expanduser()
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/so101_leader")
MIN_GOOD_FRAMES = 20

BOLD = "\033[1m"; DIM = "\033[2m"; GRN = "\033[32m"; YEL = "\033[33m"; RED = "\033[31m"; CYN = "\033[36m"; RST = "\033[0m"


def line(c="─"):
    print(c * 64)


def gap(n=2):
    for _ in range(n):
        print()


def flush_stdin():
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def ask(prompt):
    flush_stdin()
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def pinned_cam(pin_file: Path, role: str, find_flag: str) -> str:
    if not pin_file.exists():
        sys.exit(
            f"{role} camera not configured — run:\n"
            f"  python {ROOT}/eval/cam_check.py {find_flag}"
        )
    p = pin_file.read_text().strip()
    real = os.path.realpath(p)
    if not os.path.exists(real):
        sys.exit(
            f"{role} camera path {p} is gone — replug/remount it, or re-run "
            f"cam_check.py {find_flag}"
        )
    return real


def cam_config(path: str, role: str) -> OpenCVCameraConfig:
    """Per-role size/fourcc via env (e.g. FRONT_WIDTH=640 FRONT_FOURCC=YUYV)."""
    prefix = role.upper()
    width = int(os.environ.get(f"{prefix}_WIDTH", os.environ.get("CAM_WIDTH", "1280")))
    height = int(os.environ.get(f"{prefix}_HEIGHT", os.environ.get("CAM_HEIGHT", "720")))
    fps = int(os.environ.get(f"{prefix}_FPS", os.environ.get("CAM_FPS", "30")))
    fourcc = os.environ.get(f"{prefix}_FOURCC", os.environ.get("CAM_FOURCC", "MJPG"))
    return OpenCVCameraConfig(
        index_or_path=path, width=width, height=height, fps=fps, fourcc=fourcc
    )


cameras = {
    "front": cam_config(pinned_cam(FRONT_CAM, "front", "--find-front"), "front"),
}
if WRIST:
    wrist_real = pinned_cam(WRIST_CAM, "wrist", "--find")
    if wrist_real == cameras["front"].index_or_path:
        sys.exit(
            f"front and wrist are pinned to the SAME device ({wrist_real}) — "
            "UVC cams are single-reader, so this cannot record. Re-pin one of them."
        )
    cameras["wrist"] = cam_config(wrist_real, "wrist")

robot_cfg = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id="follower",
    cameras=cameras,
)
teleop_cfg = SO101LeaderConfig(port=LEADER_PORT, id="leader")
robot = SO101Follower(robot_cfg)
teleop = SO101Leader(teleop_cfg)

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
features = {**action_features, **obs_features}

WRITER_THREADS = 4 * len(cameras)
resuming = (ROOT_DS / "meta").is_dir()
if resuming and not (ROOT_DS / "meta" / "tasks.parquet").exists():
    sys.exit(
        f"\nDataset dir {ROOT_DS} exists but holds no saved episodes (aborted session).\n"
        f"Delete it and re-run:   rm -rf {ROOT_DS}"
    )
if resuming:
    dataset = LeRobotDataset.resume(REPO_ID, root=str(ROOT_DS), image_writer_threads=WRITER_THREADS)
    already = dataset.meta.total_episodes
    have = {k for k in dataset.features if k.startswith("observation.images.")}
    want = {f"observation.images.{c}" for c in cameras}
    if have != want:
        sys.exit(
            f"\nDataset at {ROOT_DS} was recorded with cameras {sorted(have)}\n"
            f"but this session is configured for {sorted(want)}.\n"
            f"Use a fresh DATASET_ROOT for a new observation space, or set WRIST=0/1 to match."
        )
else:
    ROOT_DS.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        REPO_ID,
        fps=FPS,
        features=features,
        root=str(ROOT_DS),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=WRITER_THREADS,
    )
    already = 0

if os.environ.get("REC_SELFTEST") == "1":
    print("REC_SELFTEST ok")
    print("  dataset :", ROOT_DS, f"(repo {REPO_ID}, resuming={resuming}, episodes={already})")
    print("  cameras :", {k: v.index_or_path for k, v in cameras.items()})
    print("  features:", sorted(features))
    if not resuming:
        import shutil

        shutil.rmtree(ROOT_DS, ignore_errors=True)
    sys.exit(0)

teleop_ap, robot_ap, robot_op = make_default_processors()
events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def run_phase(record: bool):
    events["exit_early"] = False
    err = {}

    def worker():
        try:
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_ap,
                robot_action_processor=robot_ap,
                robot_observation_processor=robot_op,
                teleop=teleop,
                dataset=dataset if record else None,
                control_time_s=6000,
                single_task=TASK,
                display_data=False,
            )
        except Exception as e:
            err["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t0 = time.perf_counter()
    flush_stdin()
    t.start()
    input()
    events["exit_early"] = True
    t.join()
    return time.perf_counter() - t0, err.get("e")


def pending_frames():
    try:
        return dataset.writer.episode_buffer["size"] if dataset.has_pending_frames() else 0
    except Exception:
        return 1 if dataset.has_pending_frames() else 0


if TARGET_ENV.isdigit() and int(TARGET_ENV) > 0:
    TARGET = int(TARGET_ENV)
else:
    TARGET = None
    while TARGET is None:
        try:
            _r = input("How many demos do you want to record this session? [10]: ").strip()
        except EOFError:
            _r = ""
        if _r == "":
            TARGET = 10
        elif _r.isdigit() and int(_r) > 0:
            TARGET = int(_r)
        else:
            print("  please enter a positive whole number (e.g. 20).")

print()
line("━")
print(f"{BOLD} GUIDED PICK-AND-PLACE RECORDER{RST}")
line("━")
print(f" Task        : {CYN}{TASK}{RST}")
print(f" Dataset     : {ROOT_DS}")
print(f" Already saved: {GRN}{already}{RST} good episode(s){'  (resuming)' if resuming else '  (new)'}")
print(f" Goal this run: record {BOLD}{TARGET}{RST} more demo(s)")
for _cn, _cc in cameras.items():
    print(f" Camera      : {_cn} = {_cc.index_or_path}")
if not WRIST:
    print(f"{YEL} (front-only session — set WRIST=1 for dual-cam v2 space){RST}")

gap()
print(f"{BOLD} DEMO PROTOCOL:{RST}")
print("  • Keep the puck in roughly the SAME place each episode.")
print("  • Pick → carry → place in the bin (normal demos only — no correction trials).")
print("  • Prefer a slow, deliberate close around the puck; release in the bin.")

gap()
print(f"{YEL} SAFETY: on connect the follower ENERGIZES and snaps to the leader's pose.{RST}")
print(f"{YEL}         Match the two arms by hand and keep clear.{RST}")
print(f"{DIM} Gripper tip: hold the object with a LIGHT grip — don't squeeze it fully shut.{RST}")

gap()
if ask(f"{BOLD}Press ENTER to connect the arms{RST} (or type q to abort): ") == "q":
    sys.exit(0)

robot.connect()
teleop.connect()

gap()
print(f"{GRN}✓ Arms connected and mirroring.{RST} Move the leader — the follower follows.")

saved = 0
try:
    while saved < TARGET:
        n_total = dataset.meta.total_episodes
        gap()
        line("━")
        print(
            f"{BOLD} EPISODE {n_total + 1}   "
            f"{DIM}(demo {saved + 1} of {TARGET} this run · {n_total} saved total){RST}"
        )
        line("━")

        gap()
        print(f"{BOLD}STEP 1 · SET UP{RST}")
        print(" • Put the puck in the usual spot (same place each demo).")
        print(" • Move the LEADER arm to your start pose (follower mirrors live).")
        print(f" {DIM}• then pick → place in the bin{RST}")
        gap()
        if ask(f" {BOLD}>> Press ENTER to START recording{RST} (q = finish session): ") == "q":
            break

        gap()
        print(f"{RED}{BOLD} 🔴 RECORDING…{RST} perform the pick & place now.")
        dur, err = run_phase(record=True)
        print(f" {BOLD}>> (you pressed ENTER — recording stopped){RST}")

        nf = pending_frames()
        if err is not None:
            gap()
            print(f"{RED} ⚠ The arm faulted mid-episode ({type(err).__name__}: {err}).{RST}")
            print(f"{RED}   Likely a gripper overload. This episode will be discarded.{RST}")
            dataset.clear_episode_buffer()
            gap()
            ask(" Re-match the arms if needed, then press ENTER to try again: ")
            continue
        if nf == 0:
            gap()
            print(f"{YEL} ⚠ No frames captured — nothing to save. Let's redo this one.{RST}")
            dataset.clear_episode_buffer()
            continue
        gap()
        print(f" Captured {BOLD}{nf}{RST} frames ({dur:.1f}s).")
        if nf < MIN_GOOD_FRAMES:
            print(f"{YEL}   That's very short (<{MIN_GOOD_FRAMES / FPS:.1f}s) — probably not a full demo.{RST}")

        gap()
        while True:
            c = ask(f"{BOLD}STEP 3 · KEEP THIS DEMO?{RST}  [ENTER]=keep  ·  r=redo  ·  q=save & quit: ")
            if c in ("", "k", "keep"):
                dataset.save_episode()
                saved += 1
                print(f"{GRN} ✓ Saved. Good episodes total: {dataset.meta.total_episodes}{RST}")
                decision = "keep"
                break
            if c in ("r", "redo"):
                dataset.clear_episode_buffer()
                print(f"{DIM} ↺ Discarded. Re-recording this demo…{RST}")
                decision = "redo"
                break
            if c in ("q", "quit"):
                dataset.save_episode()
                saved += 1
                print(f"{GRN} ✓ Saved. Finishing session.{RST}")
                decision = "quit"
                break
            print(f"{DIM}   (just press ENTER to keep, or r, or q){RST}")

        if decision == "quit":
            break
        if decision == "redo":
            continue

        if saved < TARGET:
            gap()
            print(f"{BOLD}STEP 4 · RESET{RST} — reposition the object for the next demo.")
            gap()
            ask(f" {BOLD}>> Press ENTER when the scene is ready for the next demo{RST}: ")

    gap()
    print(f"{GRN}{BOLD}Session done.{RST} Recorded {saved} demo(s) this run.")
except KeyboardInterrupt:
    print(f"\n{YEL}Interrupted — saving what we have.{RST}")
finally:
    print("Finalizing dataset…")
    try:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
    except Exception as e:
        print(f"{YEL}  (finalize note: {e}){RST}")
    for name, dev in (("follower", robot), ("leader", teleop)):
        try:
            if dev.is_connected:
                dev.disconnect()
        except Exception as e:
            print(f"{YEL}  ({name} disconnect note: {e}){RST}")
    print(f"{GRN}✓ Dataset at {ROOT_DS} now has {dataset.meta.total_episodes} episode(s).{RST}")
