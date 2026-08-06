#!/usr/bin/env python
"""Score a recorded rollout series, episode by episode.

    python so101/eval/review.py so101/eval/run_recordings/<stamp>_<policy>
    python so101/eval/review.py <dir> --strip 8

Writes beside the dataset:

    review/summary.md          one row per episode + the success rate
    review/ep00/front.jpg      contact sheet across the episode
    review/ep00/trace.txt      joint + gripper numbers over time

Reads the parquet and the video files directly rather than going through
LeRobotDataset: a run whose episode finalisation did not complete leaves
meta/info.json saying 0 episodes while all 1891 frames and both videos are
perfectly intact, and that run still needs scoring.

The verdict combines two measured signals, both calibrated against the demos:
  * how long the gripper holds inside the demos' carry band, and
  * whether the puck actually moved from where it started.
A failed attempt still closes the jaws, so grip alone is not enough -- the
63 s failure held 2.6 s in-band while never shifting the puck at all.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import av
import cv2
import numpy as np
import pyarrow.parquet as pq


def find_puck(bgr, lo=(95, 70, 40), hi=(135, 255, 255)):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < 60:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0 or a / (np.pi * r * r) < 0.55:
            continue
        if best is None or a > best[0]:
            best = (a, x, y, r)
    return None if best is None else (best[1], best[2], best[3])


def box_zone(demo_root: Path, frames_of):
    """Where the puck ends up in the demos IS the box, so measure it instead of
    hardcoding a rectangle that would rot the moment the box is nudged."""
    pts = []
    for im in frames_of:
        p = find_puck(im)
        if p:
            pts.append((p[0], p[1]))
    if len(pts) < 5:
        return None
    P = np.array(pts)
    mx, my, sx, sy = P[:, 0].mean(), P[:, 1].mean(), P[:, 0].std(), P[:, 1].std()
    return (mx - 3 * sx, mx + 3 * sx, my - 3 * sy, my + 3 * sy)


def carry_band(demo_root: Path, gi: int) -> tuple[float, float] | None:
    """The grip plateau the demos hold while carrying (they open ~39 first, so
    peak grip is the approach, not the hold)."""
    holds = []
    for f in sorted(glob.glob(str(demo_root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(f).to_pandas()
        if "observation.state" not in t or "episode_index" not in t:
            continue
        for _e, g in t.groupby("episode_index"):
            a = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, gi]
            lo, hi = int(0.5 * len(a)), int(0.9 * len(a))
            if hi > lo:
                holds.append(float(np.median(a[lo:hi])))
    if not holds:
        return None
    h = np.array(holds)
    return float(h.mean() - 3 * h.std()), float(h.mean() + 3 * h.std())


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1]).expanduser().resolve()
    strip_n = int(sys.argv[sys.argv.index("--strip") + 1]) if "--strip" in sys.argv else 8
    so101 = Path(__file__).resolve().parents[1]
    demo_root = Path(os.environ.get("DEMO_ROOT", so101 / "datasets" / "pick_place_v3"))

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    names = [str(n).removesuffix(".pos") for n in info["features"]["observation.state"]["names"]]
    gi = names.index("gripper")
    band = carry_band(demo_root, gi)
    if band is None:
        sys.exit("could not measure the demos' carry band — set DEMO_ROOT")

    # The demos all end with the puck in the box, so their final puck positions
    # define the target zone.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dd = LeRobotDataset(f"local/{demo_root.name}", root=str(demo_root), video_backend="pyav")
    de = dd.meta.episodes
    ends = []
    for ep in range(dd.num_episodes):
        to = int(de["dataset_to_index"][ep])
        for back in (3, 10, 25):
            im = dd[to - back]["observation.images.front"]
            ends.append((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy())
            break
    BOX = box_zone(demo_root, ends)
    del dd

    # Frames, in recorded order, straight from the parquet.
    parts = [pq.read_table(f).to_pandas()
             for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))]
    if not parts:
        sys.exit(f"no data parquet under {root}")
    import pandas as pd
    t = pd.concat(parts, ignore_index=True)
    st = np.stack(t["observation.state"].to_numpy())
    ep_of = t["episode_index"].to_numpy() if "episode_index" in t else np.zeros(len(t), int)

    # Video decoded sequentially in file order; frame i lines up with row i.
    vids = sorted(glob.glob(str(root / "videos" / "observation.images.front" / "**" / "*.mp4"),
                            recursive=True))
    frames = []
    for v in vids:
        with av.open(v) as c:
            frames.extend(f.to_ndarray(format="bgr24") for f in c.decode(c.streams.video[0]))
    if len(frames) < len(t):
        print(f"  (note: {len(frames)} video frames vs {len(t)} rows — using the shorter)")

    out = root / "review"
    out.mkdir(exist_ok=True)
    rows = []
    for ep in sorted(set(ep_of.tolist())):
        sel = np.where(ep_of == ep)[0]
        sel = sel[sel < len(frames)]
        if len(sel) == 0:
            continue
        g = st[sel, gi]
        inb = (g >= band[0]) & (g <= band[1])
        run = best = 0
        for v in inb:
            run = run + 1 if v else 0
            best = max(best, run)
        held = best / fps

        # Did the puck ever reach the box? That is the task, and it survives the
        # policy picking it back out afterwards -- which it does, having no idea
        # it has finished.
        in_box = False
        if BOX:
            for i in sel[::15]:
                q = find_puck(frames[i])
                if q and BOX[0] <= q[0] <= BOX[1] and BOX[2] <= q[1] <= BOX[3]:
                    in_box = True
                    break

        p0 = find_puck(frames[sel[0]])
        p1 = find_puck(frames[sel[-1]])
        moved_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) if (p0 and p1) else None
        px_cm = (p0[2] / 3.5) if p0 else 7.1
        moved_cm = moved_px / px_cm if moved_px is not None else None

        # A real place both holds the puck and relocates it. A peck-and-retry
        # closes the jaws briefly and leaves the puck exactly where it started.
        if BOX:
            verdict = "SUCCESS - puck reached the box" if in_box else "FAIL - puck never reached the box"
        elif moved_cm is None and held < 3.0:
            # Arm parked over the puck hides it, but a run that never held the
            # puck did not place it either -- no need to defer that one.
            verdict = "FAIL - never secured the puck (puck hidden at end)"
        elif moved_cm is None:
            verdict = "puck not visible - review the sheet"
        elif held >= 3.0 and moved_cm >= 5.0:
            verdict = "SUCCESS - carried and placed"
        elif moved_cm >= 5.0:
            verdict = "moved the puck, brief hold - check the sheet"
        elif held >= 3.0:
            verdict = "held but did not relocate it"
        else:
            verdict = "FAIL - never secured the puck"

        d = out / f"ep{ep:02d}"
        d.mkdir(exist_ok=True)
        idx = sel[np.linspace(0, len(sel) - 1, strip_n).astype(int)]
        tiles = []
        for i in idx:
            im = cv2.resize(frames[i], (320, 240))
            cv2.rectangle(im, (0, 0), (319, 18), (0, 0, 0), -1)
            cv2.putText(im, f"{(i-sel[0])/fps:5.1f}s", (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(im)
        per = 4
        grid = [np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)]
        if len(grid[-1].shape) and grid[-1].shape[1] < grid[0].shape[1]:
            pad = np.zeros((240, grid[0].shape[1] - grid[-1].shape[1], 3), np.uint8)
            grid[-1] = np.hstack([grid[-1], pad])
        cv2.imwrite(str(d / "front.jpg"), np.vstack(grid))

        with (d / "trace.txt").open("w") as fh:
            fh.write(f"episode {ep}  {len(sel)} frames  {len(sel)/fps:.1f}s\n")
            fh.write(f"{'t(s)':>7}" + "".join(f"{n:>14}" for n in names) + "\n")
            for k in range(0, len(sel), max(1, len(sel) // 40)):
                fh.write(f"{k/fps:>7.1f}" + "".join(f"{v:>14.1f}" for v in st[sel[k]]) + "\n")

        rows.append((ep, len(sel) / fps, held, moved_cm, verdict))

    ok = sum(1 for r in rows if r[4].startswith("SUCCESS"))
    with (out / "summary.md").open("w") as fh:
        fh.write(f"# {root.name}\n\n")
        fh.write(f"Demo carry band {band[0]:.1f}-{band[1]:.1f} · "
                 f"**{ok}/{len(rows)} success**\n\n")
        fh.write("| ep | len | held in band | puck moved | verdict |\n|---:|---:|---:|---:|---|\n")
        for ep, sec, held, mv, v in rows:
            fh.write(f"| {ep} | {sec:.0f}s | {held:.1f}s | "
                     f"{'n/a' if mv is None else f'{mv:.1f} cm'} | {v} |\n")

    print(f"\n  {'ep':>3} {'len':>7} {'held':>7} {'moved':>9}   verdict")
    for ep, sec, held, mv, v in rows:
        print(f"  {ep:>3} {sec:>6.0f}s {held:>6.1f}s "
              f"{('n/a' if mv is None else f'{mv:.1f} cm'):>9}   {v}")
    print(f"\n  {ok}/{len(rows)} success        {out}/summary.md")


if __name__ == "__main__":
    main()
