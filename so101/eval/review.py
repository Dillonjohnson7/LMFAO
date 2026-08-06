#!/usr/bin/env python
"""Turn a recorded series into something reviewable, per episode.

    python so101/eval/review.py so101/eval/run_recordings/<stamp>_<policy>
    python so101/eval/review.py <dir> --episodes 0,3,7      # only these
    python so101/eval/review.py <dir> --strip 8             # frames per strip

For each episode it writes, beside the dataset:

    review/
      summary.md              one table: gripper, lift, travel, verdict hint
      ep00/front.jpg          a contact sheet across the episode
      ep00/wrist.jpg          the same moments from the wrist
      ep00/trace.txt          gripper + joint numbers over time

The contact sheets are ordinary images, so a person or an agent can look at
them without a rerun viewer -- .rrd cannot be read back by the installed SDK,
and a run nobody can review is one that gets scored from memory.

The numbers are hints, never a verdict. The demos carry the puck at a known
grip band, so a close that lands far below it is the "closed on air" signature
from CHECKPOINT.md; but only the pictures show whether the puck actually
reached the box.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import av
import numpy as np


def parse_args(argv: list[str]) -> tuple[Path, list[int] | None, int]:
    if len(argv) < 2:
        sys.exit(__doc__)
    root = Path(argv[1]).expanduser().resolve()
    eps, strip = None, 8
    if "--episodes" in argv:
        eps = [int(x) for x in argv[argv.index("--episodes") + 1].split(",")]
    if "--strip" in argv:
        strip = int(argv[argv.index("--strip") + 1])
    return root, eps, strip


def episode_frames(video: Path, want: list[int]) -> dict[int, np.ndarray]:
    """Decode only the wanted frame indices, sequentially (no seeking)."""
    out, want_set = {}, set(want)
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want_set:
                out[i] = fr.to_ndarray(format="bgr24")
                if len(out) == len(want_set):
                    break
    return out


def contact_sheet(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    import cv2

    h = 240
    tiles = []
    for f, lab in zip(frames, labels, strict=False):
        t = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, 18), (0, 0, 0), -1)
        cv2.putText(t, lab, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    per_row = 4
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    import cv2
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, only, strip_n = parse_args(sys.argv)
    if not (root / "meta" / "info.json").exists():
        sys.exit(f"not a LeRobot dataset: {root}")
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    names = [str(n).removesuffix(".pos")
             for n in info["features"]["observation.state"].get("names") or []]
    gi = next((i for i, n in enumerate(names) if "gripper" in n), None)

    ds = LeRobotDataset(f"local/{root.name}", root=str(root), video_backend="pyav")
    epi = ds.meta.episodes
    n_eps = ds.num_episodes
    todo = list(range(n_eps)) if only is None else [e for e in only if e < n_eps]

    # What a real carry looks like, measured from the demos rather than assumed.
    # A demo opens the jaws (~39), closes onto the puck and HOLDS a plateau
    # (~29.6) while carrying, then opens to release. So the signal is a
    # sustained grip inside that plateau -- not the peak, which is just the
    # open-to-approach and would call a successful demo a failure.
    # Walk up looking for the demos rather than assuming a fixed depth: a wrong
    # guess previously left grip_band None and the hints were emitted anyway,
    # calling successful demos "jaws never opened". Silence beats a wrong verdict.
    demo_root = None
    if os.environ.get("DEMO_ROOT"):
        demo_root = Path(os.environ["DEMO_ROOT"])
    else:
        for base in [root, *root.parents]:
            cand = base / "datasets" / "pick_place_v3"
            if (cand / "meta" / "info.json").exists():
                demo_root = cand
                break
    grip_band = None
    if gi is not None and demo_root and (demo_root / "meta" / "info.json").exists():
        import glob
        holds = []
        for f in sorted(glob.glob(str(demo_root / "data" / "**" / "*.parquet"), recursive=True)):
            t = pq.read_table(f).to_pandas()
            if "observation.state" not in t or "episode_index" not in t:
                continue
            for _e, g in t.groupby("episode_index"):
                arr = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())[:, gi]
                lo_i, hi_i = int(0.5 * len(arr)), int(0.9 * len(arr))
                if hi_i > lo_i:
                    holds.append(float(np.median(arr[lo_i:hi_i])))
        if holds:
            h = np.array(holds)
            grip_band = (float(h.mean() - 3 * h.std()), float(h.mean() + 3 * h.std()))

    out_root = root / "review"
    out_root.mkdir(exist_ok=True)
    rows = []
    for ep in todo:
        fr = int(epi["dataset_from_index"][ep]); to = int(epi["dataset_to_index"][ep])
        L = to - fr
        idx = np.linspace(0, L - 1, strip_n).astype(int)

        states = np.stack([ds[fr + int(i)]["observation.state"].numpy() for i in range(L)])
        grip = states[:, gi] if gi is not None else np.zeros(L)
        gmax = float(grip.max())
        travel = float(np.abs(states[:, :5] - states[0, :5]).max())

        # Longest stretch held inside the demos' carry band, in seconds.
        held_s = 0.0
        if grip_band:
            inb = (grip >= grip_band[0]) & (grip <= grip_band[1])
            run = best = 0
            for v in inb:
                run = run + 1 if v else 0
                best = max(best, run)
            held_s = best / fps

        epdir = out_root / f"ep{ep:02d}"
        epdir.mkdir(exist_ok=True)
        for cam in ("front", "wrist"):
            key = f"observation.images.{cam}"
            if key not in info["features"]:
                continue
            frames, labs = [], []
            for i in idx:
                im = ds[fr + int(i)][key]
                frames.append((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1])
                labs.append(f"{i/fps:5.1f}s")
            if frames:
                cv2.imwrite(str(epdir / f"{cam}.jpg"), contact_sheet(frames, labs))

        with (epdir / "trace.txt").open("w") as fh:
            fh.write(f"episode {ep}  {L} frames  {L/fps:.1f}s\n")
            fh.write(f"{'t(s)':>7}" + "".join(f"{n:>14}" for n in names) + "\n")
            for i in range(0, L, max(1, L // 40)):
                fh.write(f"{i/fps:>7.1f}" + "".join(f"{v:>14.1f}" for v in states[i]) + "\n")

        if grip_band is None:
            hint = "NO REFERENCE — set DEMO_ROOT to the demos; hints suppressed"
            rows.append((ep, L / fps, gmax, held_s, travel, hint))
            continue
        opened = gmax >= grip_band[1]
        if held_s >= 1.0:
            hint = f"held {held_s:.1f}s in the carry band"
        elif not opened:
            hint = "jaws never opened to approach"
        elif float(grip[len(grip) // 2:].min()) < (grip_band[0] * 0.4 if grip_band else 8):
            hint = "opened then shut past the band - 'closed on air'"
        else:
            hint = "opened, no sustained hold"
        rows.append((ep, L / fps, gmax, held_s, travel, hint))

    with (out_root / "summary.md").open("w") as fh:
        fh.write(f"# {root.name}\n\n")
        if grip_band:
            fh.write(f"Demo carry-grip band: {grip_band[0]:.1f}-{grip_band[1]:.1f} "
                     f"(from {demo_root.name})\n\n")
        fh.write("| ep | len | peak grip | held in band | joint travel | hint |\n")
        fh.write("|---:|----:|----------:|-------------:|-------------:|------|\n")
        for ep, sec, g, hs, tr, hint in rows:
            fh.write(f"| {ep} | {sec:.0f}s | {g:.1f} | {hs:.1f}s | {tr:.0f} deg | {hint} |\n")
        fh.write("\nHints are signals, not verdicts — the contact sheets show whether the "
                 "puck reached the box.\n")

    print(f"review written: {out_root}")
    for ep, sec, g, hs, tr, hint in rows:
        print(f"  ep{ep:02d}  {sec:5.1f}s  peak {g:5.1f}  held {hs:4.1f}s  travel {tr:4.0f} deg   {hint}")


if __name__ == "__main__":
    main()
