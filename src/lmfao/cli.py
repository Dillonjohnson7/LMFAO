"""``lmfao`` command-line interface.

Two subcommands, both native to the LeRobot v3 on-disk format:

    lmfao inspect  <lerobot-dataset>
    lmfao generate --input <lerobot-dataset> --output <dir> --config config.json
    lmfao generate --demo --output <dir>

``inspect`` summarizes any LeRobot dataset (episodes, streams, tasks, fps, and
LMFAO provenance if present) without decoding video. ``generate`` loads real
episodes (from a LeRobot dataset, or the built-in toy scene with ``--demo``),
passes them through ``generate_training_set`` (miniworld GENERATE + pipeline
ADJUST, both driven by the JSON config), and writes the resulting training set
back out as a LeRobot-style dataset.

Config JSON is the same combined schema ``generate_training_set`` consumes::

    {
      "miniworld": {"enabled": true, "n_synthetic": 6, "camera_offsets": [...],
                    "object_pose_region": [[...],[...]], "seed": 7},
      "pipeline":  [{"name": "lighting.brightness", "params": {}, "probability": 0.5}]
    }

Configs are validated up front, before any dataset I/O, so a typo fails in
milliseconds instead of after a multi-gigabyte load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lmfao.datasets import Episode, read_lerobot_dataset, write_lerobot_dataset
from lmfao.datasets.poses import assume_camera_track
from lmfao.miniworld import MiniWorldConfig, default_intrinsics, look_at
from lmfao.pipeline import AugmentationPipeline
from lmfao.program import generate_training_set


class CliError(Exception):
    """A user-facing error: printed as one clean line, no traceback."""


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
    # Even dimensions so the result stays encodable (libx264/yuv420p).
    nh = max(2, round(h * scale)) // 2 * 2
    nw = max(2, round(w * scale)) // 2 * 2
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
    """Parse and fully validate the config before any dataset I/O happens."""
    if path is None:
        return {"miniworld": {"enabled": False}, "pipeline": []}
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise CliError(f"cannot read config {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise CliError(f"config {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise CliError("config must be a JSON object with 'miniworld' and/or 'pipeline'")
    unknown = set(data) - {"miniworld", "pipeline"}
    if unknown:
        raise CliError(
            f"unknown top-level config keys: {', '.join(sorted(unknown))} "
            "(expected 'miniworld' and/or 'pipeline')"
        )
    pipeline = data.get("pipeline", []) or []
    if not isinstance(pipeline, list):
        raise CliError("config 'pipeline' must be a list of augmentation steps")
    try:
        if data.get("miniworld"):
            MiniWorldConfig.from_dict(data["miniworld"])
        AugmentationPipeline.from_config(pipeline)
    except (ValueError, TypeError, KeyError) as e:
        raise CliError(f"invalid config {path}: {e}") from e
    return data


def _miniworld_cfg(config: dict) -> MiniWorldConfig:
    block = config.get("miniworld")
    return MiniWorldConfig.from_dict(block) if block else MiniWorldConfig(enabled=False)


def _generate(args: argparse.Namespace) -> int:
    if args.seed is not None and args.seed < 0:
        raise CliError("--seed must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise CliError("--limit must be non-negative")
    if args.max_frames is not None and args.max_frames < 1:
        raise CliError("--max-frames must be at least 1")

    config = _load_config(args.config)
    mw = _miniworld_cfg(config)
    wants_generate = mw.enabled and mw.n_synthetic > 0

    out = Path(args.output)
    if (out / "meta" / "info.json").exists() and not args.overwrite:
        raise CliError(
            f"{out} already contains a dataset; pass --overwrite to replace it"
        )

    if args.demo:
        real = _demo_episodes()
        print(f"loaded {len(real)} demo episodes (toy pick_place scene)")
    else:
        if not args.input:
            raise CliError("--input <lerobot-dataset> is required (or use --demo)")
        try:
            real = read_lerobot_dataset(
                args.input, video_key=args.video_key, limit=args.limit,
                max_frames=args.max_frames,
            )
        except (OSError, ValueError) as e:
            raise CliError(f"cannot read {args.input}: {e}") from e
        print(f"loaded {len(real)} episodes from {args.input}")
    if not real:
        raise CliError("no episodes loaded; nothing to do")

    # The pure-numpy reference renderer is impractical at full camera resolution,
    # so GENERATE runs at a small working size on real footage.
    if wants_generate and not args.demo and args.work_size > 0:
        before = (real[0].height, real[0].width)
        real = [_downscale_episode(ep, args.work_size) for ep in real]
        if (real[0].height, real[0].width) != before:
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

    try:
        write_lerobot_dataset(training_set.episodes, out, video_key=args.write_video_key)
    except (OSError, ValueError) as e:
        raise CliError(f"cannot write {out}: {e}") from e
    print(f"wrote {summary['episodes']} episodes -> {out}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    """Summarize a LeRobot dataset without decoding any video."""
    root = Path(args.dataset)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise CliError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise CliError(
            "inspect needs pyarrow; install the extra: pip install 'lmfao[lerobot]'"
        ) from e

    from lmfao.datasets.lerobot import _episode_records, _read_tasks, _video_keys

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CliError(f"cannot read {info_path}: {e}") from e
    records = _episode_records(root, pq)
    tasks = _read_tasks(root, pq)
    keys = _video_keys(info)
    on_disk = {k for k in keys if (root / "videos" / k).exists()}

    print(f"dataset:   {root}")
    print(f"version:   {info.get('codebase_version', '?')}   "
          f"robot: {info.get('robot_type', '?')}   fps: {info.get('fps', '?')}")
    print(f"episodes:  {len(records)} on disk "
          f"(info.json declares {info.get('total_episodes', '?')}, "
          f"{info.get('total_frames', '?')} frames)")
    print(f"tasks:     {len(tasks) or len({str(r.get('tasks')) for r in records})}")
    for k in keys:
        marker = "ok" if k in on_disk else "MISSING on disk"
        print(f"video:     {k} [{marker}]")
    missing_data = sorted(
        {
            str(root / info["data_path"].format(
                chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])
            ))
            for r in records
            if not (root / info["data_path"].format(
                chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])
            )).exists()
        }
    ) if records and "data_path" in info else []
    for p in missing_data:
        print(f"warning:   data parquet MISSING: {p} (partial download?)")

    prov_path = root / "meta" / "lmfao_provenance.json"
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CliError(f"cannot read {prov_path}: {e}") from e
        synth = sum(1 for p in prov if p.get("synthetic"))
        print(f"lmfao:     provenance sidecar present — "
              f"{synth} synthetic / {len(prov) - synth} real episode(s)")

    if args.episodes:
        for r in records:
            ei = int(r["episode_index"])
            task = r.get("tasks", [""])
            task = task[0] if isinstance(task, list) and task else ""
            print(f"  episode {ei}: {int(r['length'])} frames  task={task!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmfao", description="LMFAO data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("inspect", help="summarize a LeRobot dataset (no video decode)")
    ins.add_argument("dataset", help="path to a LeRobot dataset root")
    ins.add_argument("--episodes", action="store_true", help="also list every episode")
    ins.set_defaults(func=_inspect)

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
    gen.add_argument("--overwrite", action="store_true",
                     help="replace an existing dataset at --output instead of erroring")
    gen.set_defaults(func=_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
