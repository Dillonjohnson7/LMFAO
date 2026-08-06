#!/usr/bin/env python
"""
Camera bring-up & wrist-cam discovery for the SO101 workcell.

  python cam_check.py                 Test both pinned cameras: open, grab frames,
                                      measure achieved fps, save a probe JPEG per
                                      camera to eval/cam_probe/, and list which
                                      cameras the Anvil ROS2 stack is holding.
                                      No robot involved — always safe.
  python cam_check.py --find          GUIDED wrist-cam discovery: unplug it, plug it
                                      into its final DIRECT USB port, and this pins
                                      the new device's stable /dev/v4l/by-path name
                                      into wrist_cam.path.
  python cam_check.py --find-front    Same guided flow for the front cam
                                      (-> front_cam.path).
  python cam_check.py --set DEV       Pin DEV (e.g. /dev/video10) as the wrist cam
                                      without the guided flow.
  python cam_check.py --set-front DEV Pin DEV as the front cam.

Both roles are pinned explicitly because camera<->role is a physical mounting
decision no probe can discover — and an image alone will not tell you: with the
arm rotated, the fixed front cam frames little but the gripper and table, which
looks just like a wrist view (CHECKPOINT.md §1).

Why by-path and not by-id: the workcell has several IDENTICAL no-serial UVC
cams (Anvil's), so /dev/v4l/by-id names collide; /dev/videoN renumbers across
reboots. /dev/v4l/by-path is stable per PHYSICAL USB PORT — and the wrist cam
lives on a fixed port + fixed mount. (Same lesson as the arms' udev rules.)

⚠ The Anvil workcell ROS2 stack (container anvil-loader-ros2-1, privileged) runs
one usb_cam node per workcell camera and STREAMS them. udev pins those cameras
by PCI address and publishes the map as symlinks in /run/cameras (cam_waist,
cam_chest, cam_wrist_r, cam_wrist_l) — this tool reads that map live and warns
before letting you pin one of them.

⚠ A camera someone else is streaming looks perfectly healthy: it open()s fine,
answers VIDIOC_QUERYCAP, and `fuser`/`lsof` report it FREE, because the holder
is in a container namespace. The only honest signal is VIDIOC_S_FMT -> EBUSY,
which stream_busy() below probes directly. (Cost a debugging cycle on 07-21 —
CHECKPOINT.md Mistake #15.)
"""
import errno
import fcntl
import os
import struct
import subprocess
import sys
import time

import cv2

# Paths live under LMFAO/so101 (override via SO101_ROOT / *_CAM_FILE).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT as _ROOT, FRONT_CAM_FILE as _FCF, WRIST_CAM_FILE as _WCF  # noqa: E402
BASE = str(_ROOT)
WRIST_CAM_FILE = str(_WCF)
FRONT_CAM_FILE = str(_FCF)
PROBE_DIR = str(_ROOT / "eval" / "cam_probe")
# No hardcoded default for either role: the Sonix that was the v1 front cam is
# the v2 WRIST cam (remounted 07-21 without a USB replug). Roles come only from
# the pin files; every v2 tool follows the pins.
# Cameras the Anvil workcell ROS2 stack owns. Read live from the udev-maintained
# symlink dir rather than hardcoding node numbers, which renumber across replugs.
ANVIL_CAM_DIR = "/run/cameras"
ROS2_CONTAINER = "anvil-loader-ros2"          # substring match against `docker ps`
W, H, FPS = 1280, 720, 30
# Per-role capture geometry, from the pick_place_v3 training data (and so the
# policies trained on it): front is 640x480, wrist is 1280x720, both at 30 fps.
# Probing every camera at 1280x720 falsely fails a front cam that is correct —
# the RealSense D415 color stream has no 30 fps mode at 720p (uncompressed YUYV
# only) but does deliver 30 fps at the 640x480 it is actually used at.
ROLE_GEOM = {"front": (640, 480), "wrist": (1280, 720)}

# VIDIOC_G_FMT / VIDIOC_S_FMT — _IOWR('V', 4|5, struct v4l2_format[208])
VIDIOC_G_FMT = (3 << 30) | (208 << 16) | (ord("V") << 8) | 4
VIDIOC_S_FMT = (3 << 30) | (208 << 16) | (ord("V") << 8) | 5
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1

BOLD = "\033[1m"; GRN = "\033[32m"; YEL = "\033[33m"; RED = "\033[31m"; RST = "\033[0m"


def busy_pids(dev):
    """PIDs currently holding a /dev/video node (empty string if free)."""
    try:
        return subprocess.run(["fuser", dev], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def video_nodes():
    return sorted(d for d in os.listdir("/dev") if d.startswith("video"))


def anvil_cams():
    """{'/dev/videoN': 'cam_waist', ...} — cameras the Anvil workcell ROS2 stack owns.

    Source of truth is /run/cameras, maintained by /etc/udev/rules.d/99-camera.rules
    (one PCI address per camera). Empty dict if the workcell isn't set up.
    """
    out = {}
    if not os.path.isdir(ANVIL_CAM_DIR):
        return out
    for name in sorted(os.listdir(ANVIL_CAM_DIR)):
        real = os.path.realpath(os.path.join(ANVIL_CAM_DIR, name))
        if os.path.exists(real):
            out[real] = name
    return out


def ros2_stack_up():
    """Name of the running Anvil ROS2 container (it streams the workcell cams), or None."""
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=5).stdout
        return next((n for n in out.split() if ROS2_CONTAINER in n), None)
    except Exception:
        return None


def stream_busy(dev):
    """True if something is already STREAMING from dev (VIDIOC_S_FMT -> EBUSY).

    Why not fuser/lsof: the holder is normally a usb_cam node inside the privileged
    anvil-loader-ros2-1 container and is invisible from the host. A taken camera
    still open()s and answers QUERYCAP, so only S_FMT tells the truth.
    Non-destructive: we read the current format back and write the same one.
    """
    try:
        fd = os.open(os.path.realpath(dev), os.O_RDWR)
    except OSError:
        return False
    try:
        buf = bytearray(208)
        struct.pack_into("<I", buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, VIDIOC_G_FMT, buf, True)
        fcntl.ioctl(fd, VIDIOC_S_FMT, buf, True)
        return False
    except OSError as e:
        return e.errno == errno.EBUSY
    finally:
        os.close(fd)


def explain_busy(dev):
    """Print why `dev` is unusable, naming the real holder."""
    owned = anvil_cams().get(os.path.realpath(dev))
    holder = ros2_stack_up()
    print(f"  {RED}✗ {dev} is BUSY — something else is already streaming it.{RST}")
    if owned:
        print(f"    It is the Anvil workcell's {YEL}{owned}{RST} (/run/cameras/{owned}).")
    if holder:
        print(f"    Holder is almost certainly a usb_cam node in {YEL}{holder}{RST} "
              f"(privileged — holds every /dev/video*).")
    print(f"    {YEL}fuser/lsof will look clean: the holder is in a container namespace.{RST}")


def by_path_for(node):
    """Stable /dev/v4l/by-path symlink resolving to `node` (prefer index0)."""
    d = "/dev/v4l/by-path"
    if not os.path.isdir(d):
        return None
    hits = [os.path.join(d, s) for s in sorted(os.listdir(d))
            if os.path.realpath(os.path.join(d, s)) == os.path.realpath(node)]
    for h in hits:
        if h.endswith("index0"):
            return h
    return hits[0] if hits else None


def grab_test(dev, label, n=45):
    """Open dev at its role's trained geometry (MJPG, 30 fps), read n frames,
    report. True on success."""
    w_want, h_want = ROLE_GEOM.get(label, (W, H))
    real = os.path.realpath(dev)              # cv2's V4L2 backend can't open by-id/by-path symlinks
    print(f"  {label}: opening {dev}" + (f" -> {real}" if real != dev else "") + " ...", flush=True)
    cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_want)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_want)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print(f"  {RED}✗ {label}: could not open {dev}{RST}")
        return False
    ok_any, frame = False, None
    for _ in range(5):                       # warm-up
        ok_any, frame = cap.read()
    t0 = time.perf_counter()
    got = 0
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            got += 1
            frame = f
    dt = time.perf_counter() - t0
    cap.release()
    if not ok_any or frame is None or got == 0:
        print(f"  {RED}✗ {label}: opened but returned no frames{RST}")
        return False
    h, w = frame.shape[:2]
    fps = got / dt if dt > 0 else 0
    os.makedirs(PROBE_DIR, exist_ok=True)
    out = f"{PROBE_DIR}/{label}.jpg"
    cv2.imwrite(out, frame)
    good = (w, h) == (w_want, h_want) and fps > 25
    mark = f"{GRN}✓{RST}" if good else f"{YEL}⚠{RST}"
    print(f"  {mark} {label}: {w}x{h} @ {fps:.1f} fps ({got}/{n} frames) — probe saved to {out}")
    if (w, h) != (w_want, h_want):
        print(f"    {YEL}resolution is not {w_want}x{h_want} — the recorder/policy "
              f"expect {w_want}x{h_want} for {label}{RST}")
    if fps <= 25:
        print(f"    {YEL}below 30 fps — try a different (direct) USB port{RST}")
    return good


def wrist_path():
    if not os.path.exists(WRIST_CAM_FILE):
        return None
    return open(WRIST_CAM_FILE).read().strip()


def front_path():
    if not os.path.exists(FRONT_CAM_FILE):
        return None
    return open(FRONT_CAM_FILE).read().strip()


def pin(dev, role="wrist"):
    """Resolve dev to by-path, warn about conflicts, confirm, write the role's pin."""
    pin_file = WRIST_CAM_FILE if role == "wrist" else FRONT_CAM_FILE
    real = os.path.realpath(dev)
    if not os.path.exists(real):
        sys.exit(f"{RED}{dev} does not exist{RST}")
    owned = anvil_cams().get(real)
    if owned:
        print(f"\n{YEL}⚠ {real} is the Anvil workcell's {owned} (/run/cameras/{owned}), "
              f"pinned there by /etc/udev/rules.d/99-camera.rules. Taking it means that "
              f"stack loses the camera — and a `docker restart` of the ROS2 container "
              f"would take it back, mid-session.{RST}")
        if input("  Type YES to pin it anyway, anything else to abort: ").strip() != "YES":
            sys.exit("aborted — use a port with no /run/cameras rule (a motherboard port)")
    if stream_busy(real):
        explain_busy(real)
        sys.exit("aborted — free the camera or move it to another port, then re-run")
    pids = busy_pids(real)            # host-side holders; blind to containers, hence the above
    if pids:
        print(f"\n{YEL}⚠ {real} is currently held by PID(s) {pids} — recording will fail "
              f"while that process reads it.{RST}")
        if input("  Type YES to pin it anyway, anything else to abort: ").strip() != "YES":
            sys.exit("aborted")
    stable = by_path_for(real)
    if stable is None:
        print(f"{YEL}⚠ no /dev/v4l/by-path symlink found for {real}; pinning the raw node "
              f"(may renumber after a reboot — re-run --find if it does).{RST}")
        stable = real
    if not grab_test(stable, role):
        sys.exit(f"{RED}capture test failed — not pinning. Fix the connection and re-run.{RST}")
    with open(pin_file, "w") as f:
        f.write(stable + "\n")
    print(f"\n{GRN}✓ {role} cam pinned:{RST} {stable}\n  (written to {pin_file})")
    print(f"  Check eval/cam_probe/{role}.jpg — it should show the {role}-cam view.")


def find_flow(role="wrist"):
    print(f"{BOLD}{role.upper()}-CAM DISCOVERY{RST}")
    print(f" 1. UNPLUG the camera you're using as the {role} cam (if plugged in).")
    input("    ENTER when it is unplugged: ")
    before = {os.path.realpath(f"/dev/{n}") for n in video_nodes()}
    print(f" 2. Mount it, then plug its USB into a DIRECT PC port — a motherboard")
    print("    port, not a hub (the hub lesson, STATUS.md) and NOT one of the PCI")
    print("    slots in /etc/udev/rules.d/99-camera.rules, or the Anvil ROS2 stack")
    print("    will reclaim it on its next restart.")
    input("    ENTER once it is plugged in: ")
    print("    waiting for the new device ", end="", flush=True)
    new = []
    for _ in range(30):
        time.sleep(1)
        now = {os.path.realpath(f"/dev/{n}") for n in video_nodes()}
        # numeric sort: sorted() puts /dev/video10 before /dev/video9
        new = sorted(now - before, key=lambda p: int("".join(c for c in p if c.isdigit()) or 0))
        if new:
            break
        print(".", end="", flush=True)
    print()
    if not new:
        sys.exit(f"{RED}no new /dev/video node appeared in 30 s — check the cable/port and re-run{RST}")
    dev = new[0]                                # index0 (capture) node enumerates first
    print(f"  new device: {', '.join(new)} -> using {dev}")
    pin(dev, role)


def main():
    for flag, role in (("--find", "wrist"), ("--find-front", "front")):
        if flag in sys.argv:
            find_flow(role)
            return
    for flag, role in (("--set", "wrist"), ("--set-front", "front")):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 >= len(sys.argv):
                sys.exit(f"usage: cam_check.py {flag} /dev/videoN")
            pin(sys.argv[i + 1], role)
            return
    print(f"{BOLD}CAMERA CHECK{RST} (no robot, no motion)")
    try:
        up = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                            capture_output=True, text=True, timeout=5).stdout
        if any(n.startswith("so101-") for n in up.split()):
            print(f"{YEL}⚠ ALAN sentinel containers are running — they hold the arm.{RST}")
            print(f"{YEL}    docker stop so101-camera so101-sentinel-web{RST}")
    except Exception:
        pass
    owned = anvil_cams()
    if owned:
        holder = ros2_stack_up()
        state = f"streamed by {holder}" if holder else "stack not running"
        print(f"  Anvil workcell cameras ({state}) — do not take these:")
        for node, name in sorted(owned.items(), key=lambda kv: kv[1]):
            busy = f" {RED}BUSY{RST}" if stream_busy(node) else f" {GRN}free{RST}"
            print(f"    {name:<12} {node}{busy}")
    print()

    ok = True
    for role, path in (("front", front_path()), ("wrist", wrist_path())):
        if path is None:
            print(f"  {YEL}{role}: not pinned — run  python {BASE}/eval/cam_check.py "
                  f"--find{'' if role == 'wrist' else '-front'}{RST}")
            ok = False
        elif not os.path.exists(path):
            print(f"  {RED}✗ {role}: pinned path {path} is missing — replug it or re-run "
                  f"--find{'' if role == 'wrist' else '-front'}{RST}")
            ok = False
        elif stream_busy(path):
            explain_busy(path)
            ok = False
        else:
            ok = grab_test(path, role) and ok
    print(f"\n{'ALL GOOD — probe images in ' + PROBE_DIR if ok else 'ISSUES FOUND — see above'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
