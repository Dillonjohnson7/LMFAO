#!/usr/bin/env python
"""Render ONE episode of a pick_place dataset to a Firefox-playable H.264 mp4.

Usage: python view_episode.py [episode_index]   (default 0)
Multi-camera datasets (front + wrist) render side-by-side in one video.
Env: DATASET_ROOT (default so101/datasets/pick_place_v3).
"""
from __future__ import annotations

import os
import sys

import av
import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from _paths import DEFAULT_DATASET, DATASETS

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
_v3 = DEFAULT_DATASET
_v2 = DATASETS / "pick_place_v2"
_v1 = DATASETS / "pick_place"
if "DATASET_ROOT" in os.environ:
    ROOT = os.environ["DATASET_ROOT"]
elif (_v3 / "meta").is_dir():
    ROOT = str(_v3)
elif (_v2 / "meta").is_dir():
    ROOT = str(_v2)
else:
    ROOT = str(_v1)
REPO_ID = "local/" + os.path.basename(ROOT.rstrip("/"))

ds = LeRobotDataset(REPO_ID, root=ROOT, episodes=[EP], video_backend="pyav")
n = len(ds)
if n == 0:
    print(f"episode {EP}: no frames")
    sys.exit(1)
fps = int(ds.meta.fps)
IMG_KEYS = sorted(k for k in ds.meta.features if k.startswith("observation.images."))
print(f"dataset {ROOT} · cameras: {[k.split('.')[-1] for k in IMG_KEYS]}")


def to_rgb(t):
    a = t.numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.dtype != "uint8":
        a = (a * 255).clip(0, 255).astype("uint8")
    return np.ascontiguousarray(a)


def frame_at(i):
    s = ds[i]
    tiles = []
    for k in IMG_KEYS:
        img = to_rgb(s[k])
        cv2.putText(
            img,
            k.split(".")[-1],
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(img)
    h = min(t.shape[0] for t in tiles)
    tiles = [
        t if t.shape[0] == h else cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
        for t in tiles
    ]
    return np.ascontiguousarray(np.hstack(tiles))


first = frame_at(0)
h, w = first.shape[:2]
w -= w % 2
h -= h % 2
outdir = os.path.join(ROOT, "_review")
os.makedirs(outdir, exist_ok=True)
outp = os.path.join(outdir, f"episode_{EP}.mp4")

container = av.open(outp, mode="w")
stream = container.add_stream("libx264", rate=fps)
stream.width, stream.height = w, h
stream.pix_fmt = "yuv420p"
stream.options = {"crf": "23", "preset": "veryfast"}

for i in range(n):
    rgb = frame_at(i)[:h, :w]
    cv2.putText(
        rgb,
        f"episode {EP}   frame {i + 1}/{n}   ({(i + 1) / fps:.1f}s)",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    vframe = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    for pkt in stream.encode(vframe):
        container.mux(pkt)
    if (i + 1) % 150 == 0:
        print(f"  {i + 1}/{n} frames…")

for pkt in stream.encode():
    container.mux(pkt)
container.close()
print(f"wrote {outp}  ({n} frames, {n / fps:.1f}s, H.264)")
