"""Augment every video in a folder and write the results elsewhere.

    python examples/augment_folder.py test_videos augmented_videos
    python examples/augment_folder.py test_videos out -a noise.gaussian lighting.brightness

Augmentations are named `category.preset` and applied in the order given. Any
registered feature works here, so adding lighting or occlusion needs no change
to this script.

Needs `opencv-python` for decoding and `ffmpeg` on PATH for encoding.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import lmfao

BATCH = 32


def encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """An ffmpeg process accepting raw BGR frames on stdin."""

    # VideoToolbox is a large speedup on Macs and is always present there.
    codec = "h264_videotoolbox" if platform.system() == "Darwin" else "libx264"
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-c:v", codec, "-b:v", "6M", "-pix_fmt", "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def batches(capture: cv2.VideoCapture, size: int):
    """Yield stacked frame batches until the capture runs dry."""

    batch = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        batch.append(frame)
        if len(batch) == size:
            yield np.stack(batch)
            batch = []

    if batch:
        yield np.stack(batch)


def augment_file(source: Path, destination: Path, pipeline: lmfao.AugmentationPipeline) -> int:
    """Stream one video through ``pipeline``, returning the frame count."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")

    ffmpeg = encoder(
        destination,
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        capture.get(cv2.CAP_PROP_FPS) or 30.0,
    )

    frames = 0
    for batch in batches(capture, BATCH):
        augmented, _ = pipeline(batch)
        ffmpeg.stdin.write(np.ascontiguousarray(augmented).tobytes())
        frames += len(batch)

    capture.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("-a", "--augment", nargs="+", default=["noise.gaussian"])
    args = parser.parse_args()

    pipeline = lmfao.AugmentationPipeline.from_config([{"name": name} for name in args.augment])
    videos = sorted(args.source.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 files in {args.source}")
    args.destination.mkdir(parents=True, exist_ok=True)

    total = 0
    started = time.perf_counter()
    for video in videos:
        frames = augment_file(video, args.destination / video.name, pipeline)
        total += frames
        print(f"{video.name:<16} {frames:>6} frames", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n{total:,} frames from {len(videos)} videos in {elapsed:.1f}s -> {total / elapsed:.0f} fps")


if __name__ == "__main__":
    main()
