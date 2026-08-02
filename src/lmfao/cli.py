"""``lmfao`` command-line interface.

Today it exposes the GENERATE half of the program end-to-end:

    lmfao generate --input <lerobot-dataset> --output <dir> --config config.json
    lmfao generate --demo --output <dir>

A run loads real episodes (from a LeRobot dataset, or the built-in toy scene
with ``--demo``), passes them through ``generate_training_set`` (miniworld
GENERATE + pipeline ADJUST, both driven by the JSON config), and writes the
resulting training set back out as a LeRobot-style dataset.

Config JSON is the same combined schema ``generate_training_set`` consumes::

    {
      "miniworld": {"enabled": true, "n_synthetic": 6, "camera_offsets": [...],
                    "object_pose_region": [[...],[...]], "seed": 7},
      "pipeline":  [{"name": "lighting.brightness", "params": {}, "probability": 0.5}]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lmfao.datasets import Episode, read_lerobot_dataset, write_lerobot_dataset
from lmfao.datasets.poses import assume_camera_track
from lmfao.miniworld import default_intrinsics, look_at
from lmfao.program import generate_training_set


def _demo_episodes(count: int = 3, frames: int = 6, size: int = 48) -> list[Episode]:
    """The toy pick_place scene from examples/miniworld_usage.py (has real poses)."""
    episodes = []
    for seed in range(count):
        k = default_intrinsics(size, size, 60.0)
        imgs = np.zeros((frames, size, size, 3), np.uint8)
        poses = np.zeros((frames, 4, 4))
        yy, xx = np.mgrid[0:size, 0:size]
        for i in range(frames):
            ang = -0.3 + 0.6 * i / (frames - 1)
            poses[i] = look_at(np.array([np.sin(ang) * 0.3, -0.5 + 0.1 * i, 1.1]), np.zeros(3))
            imgs[i, ..., 0] = (xx * 6) % 256
            imgs[i, ..., 1] = (yy * 6) % 256
            imgs[i, ..., 2] = 90
            blob = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 < 20
            imgs[i][blob] = (210, 40, 40)
        state = np.tile([0.0, 0.0, 0.2, 30.0], (frames, 1)).astype(float)
        episodes.append(
            Episode(
                frames=imgs, state=state, camera_poses=poses, intrinsics=k,
                task="pick_place_v2", metadata={"episode_id": seed},
            )
        )
    return episodes


def _downscale_episode(ep: Episode, max_size: int) -> Episode:
    """Nearest-neighbour downscale an episode's frames so the longest side is
    at most ``max_size``. The pure-numpy miniworld renderer is O(gaussians) per
    pixel, so full 640px footage is impractical; a small working resolution keeps
    the reference GENERATE path interactive (the real splat backend handles full
    res). Returns the episode unchanged if it already fits or ``max_size<=0``.
    """
    if max_size <= 0:
        return ep
    h, w = ep.height, ep.width
    if max(h, w) <= max_size:
        return ep
    scale = max_size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(nw) * w / nw).astype(int)
    frames = ep.frames[:, ys][:, :, xs]
    meta = dict(ep.metadata)
    meta["work_size"] = [nh, nw]
    return Episode(
        frames=np.ascontiguousarray(frames),
        state=ep.state,
        actions=ep.actions,
        fps=ep.fps,
        task=ep.task,
        camera_poses=ep.camera_poses,
        intrinsics=ep.intrinsics,
        metadata=meta,
    )


def _load_config(path: str | None) -> dict:
    if path is None:
        return {"miniworld": {"enabled": False}, "pipeline": []}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object with 'miniworld' and/or 'pipeline'")
    return data


def _generate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    wants_generate = bool(config.get("miniworld", {}).get("enabled", False))

    if args.demo:
        real = _demo_episodes()
        print(f"loaded {len(real)} demo episodes (toy pick_place scene)")
    else:
        if not args.input:
            print("error: --input <lerobot-dataset> is required (or use --demo)", file=sys.stderr)
            return 2
        real = read_lerobot_dataset(
            args.input, video_key=args.video_key, limit=args.limit, max_frames=args.max_frames
        )
        print(f"loaded {len(real)} episodes from {args.input}")

    # The pure-numpy reference renderer is impractical at full camera resolution,
    # so GENERATE runs at a small working size on real footage.
    if wants_generate and not args.demo and args.work_size > 0:
        before = (real[0].height, real[0].width) if real else None
        real = [_downscale_episode(ep, args.work_size) for ep in real]
        if real and (real[0].height, real[0].width) != before:
            print(f"working resolution {real[0].width}x{real[0].height} "
                  f"(downscaled from {before[1]}x{before[0]} for the reference renderer)")

    # GENERATE needs camera poses; real LeRobot data has none. Attach an assumed
    # trajectory if asked, otherwise miniworld will simply skip pose-less sources.
    if wants_generate and args.assume_poses:
        attached = 0
        for i, ep in enumerate(real):
            if ep.camera_poses is None:
                real[i] = assume_camera_track(ep)
                attached += 1
        if attached:
            print(f"attached assumed camera poses to {attached} episode(s) "
                  "(approximate — not FK-derived)")
    elif wants_generate and any(ep.camera_poses is None for ep in real):
        print("note: some episodes lack camera poses; miniworld will skip those. "
              "Pass --assume-poses to generate from them anyway (approximate).")

    training_set = generate_training_set(real, config, seed=args.seed)
    summary = training_set.summary()
    print(f"training set: {summary}")

    out = Path(args.output)
    write_lerobot_dataset(training_set.episodes, out, video_key=args.write_video_key)
    print(f"wrote {summary['episodes']} episodes -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmfao", description="LMFAO data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a training set (miniworld + pipeline)")
    gen.add_argument("--input", help="path to a LeRobot dataset root")
    gen.add_argument("--output", required=True, help="output dataset directory")
    gen.add_argument("--config", help="combined {miniworld, pipeline} JSON config")
    gen.add_argument("--demo", action="store_true", help="use the built-in toy scene instead of --input")
    gen.add_argument("--seed", type=int, default=None, help="base seed")
    gen.add_argument("--video-key", default=None, help="which camera stream to read")
    gen.add_argument("--write-video-key", default="observation.images.render",
                     help="video key to write in the output dataset")
    gen.add_argument("--limit", type=int, default=None, help="load at most N episodes")
    gen.add_argument("--max-frames", type=int, default=None, help="truncate each episode to N frames")
    gen.add_argument("--assume-poses", action="store_true",
                     help="attach an assumed camera trajectory so GENERATE runs on pose-less data")
    gen.add_argument("--work-size", type=int, default=96,
                     help="max frame side for the reference GENERATE renderer (0 = full res)")
    gen.set_defaults(func=_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
