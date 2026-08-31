"""``lmfao`` command-line interface.

Run ``lmfao`` with no arguments for an interactive wizard (paste a Hugging Face
link, pick what to do). The flag-driven subcommands below are the scriptable
interface, all native to the LeRobot v3 on-disk format:

    lmfao inspect  <lerobot-dataset>
    lmfao augment  --input <lerobot-dataset> --output <dir> --config pipeline.json
    lmfao generate --input <lerobot-dataset> --output <dir> --config config.json
    lmfao generate --demo --output <dir>

``inspect`` summarizes any LeRobot dataset (episodes, streams, tasks, fps, and
LMFAO provenance if present) without decoding video. ``augment`` runs only the
ADJUST pixel pipeline over the real frames at their native resolution -- so the
output is real, full-res footage, just seasoned -- and can emit several
independently-seasoned variants per episode. ``generate`` additionally runs the
miniworld GENERATE half (synthetic novel-view episodes) alongside ADJUST.

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
from lmfao.datasets.lerobot import LeRobotStreamingWriter
from lmfao.datasets.poses import assume_camera_track
from lmfao.miniworld import MiniWorldConfig, default_intrinsics, look_at
from lmfao.pipeline import AugmentationPipeline
from lmfao.program import (
    generate_training_set,
    iter_augment_episode,
    iter_sweep_episode,
)


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


def _load_augment_config(path: str | None) -> tuple[str, list]:
    """Load and validate an augment config. Returns ``(mode, payload)``:

    - ``("pipeline", steps)`` for a bare list of steps or ``{"pipeline": [...]}``
      -- one pipeline applied together, ``--variants`` random reseeds.
    - ``("sweep", specs)`` for ``{"sweep": [{"label", "pipeline"}, ...]}`` -- a
      deterministic magnitude sweep, one output per spec per episode.
    """
    if path is None:
        raise CliError("augment needs --config with a pipeline of augmentation steps")
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise CliError(f"cannot read config {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise CliError(f"config {path} is not valid JSON: {e}") from e

    if isinstance(data, dict) and "sweep" in data:
        specs = data["sweep"]
        if not isinstance(specs, list) or not specs:
            raise CliError("augment 'sweep' must be a non-empty list of {label, pipeline} steps")
        try:
            for spec in specs:
                AugmentationPipeline.from_config(spec["pipeline"])
        except (ValueError, TypeError, KeyError) as e:
            raise CliError(f"invalid sweep config {path}: {e}") from e
        return "sweep", specs

    if isinstance(data, list):
        steps = data
    elif isinstance(data, dict):
        if "pipeline" not in data:
            raise CliError("augment config object must contain a 'pipeline' or 'sweep' list")
        steps = data["pipeline"]
        mw = data.get("miniworld")
        if isinstance(mw, dict) and mw.get("enabled"):
            print("note: augment ignores the 'miniworld' block; "
                  "use `lmfao generate` to synthesize novel views")
    else:
        raise CliError("augment config must be a list of steps or an object with 'pipeline'/'sweep'")

    if not isinstance(steps, list) or not steps:
        raise CliError("augment needs a non-empty 'pipeline' list of augmentation steps")
    try:
        AugmentationPipeline.from_config(steps)
    except (ValueError, TypeError, KeyError) as e:
        raise CliError(f"invalid config {path}: {e}") from e
    return "pipeline", steps


def _miniworld_cfg(config: dict) -> MiniWorldConfig:
    block = config.get("miniworld")
    return MiniWorldConfig.from_dict(block) if block else MiniWorldConfig(enabled=False)


def _job_signature(mode: str, payload: list, args: argparse.Namespace) -> str:
    """Stable hash of the run config, so --resume refuses to continue a run whose
    parameters changed (which would produce an inconsistent dataset)."""
    import hashlib

    key = json.dumps({
        "mode": mode, "payload": payload, "seed": args.seed,
        "variants": args.variants, "include_original": args.include_original,
        "original_copies": getattr(args, "original_copies", 1),
        "video_key": args.video_key, "write_video_key": args.write_video_key,
        "max_frames": args.max_frames, "limit": args.limit, "demo": args.demo,
        "input": args.input,
    }, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _augment(args: argparse.Namespace) -> int:
    if args.seed is not None and args.seed < 0:
        raise CliError("--seed must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise CliError("--limit must be non-negative")
    if args.max_frames is not None and args.max_frames < 1:
        raise CliError("--max-frames must be at least 1")
    if args.variants < 1:
        raise CliError("--variants must be at least 1")
    if getattr(args, "original_copies", 1) < 1:
        raise CliError("--original-copies must be at least 1")

    mode, payload = _load_augment_config(args.config)
    resume = getattr(args, "resume", False)
    out = Path(args.output)
    ckpt = out / ".lmfao_resume.pkl"
    sig = _job_signature(mode, payload, args)

    # The output is CLEARED before the source is read (lazy reader), so writing
    # into (or over) the input would delete the very dataset we are augmenting.
    if not args.demo and args.input:
        op, ip = out.resolve(), Path(args.input).resolve()
        if op == ip or op in ip.parents or ip in op.parents:
            raise CliError(
                "--output must be a separate directory from --input; augmenting a "
                "dataset in place would delete the source"
            )
    # A stream key becomes a path component; reject anything that escapes --output.
    if args.write_video_key is not None:
        _validate_video_key(args.write_video_key)

    # --- resolve the source episodes as a lazy per-index reader (bounded memory) ---
    try:
        if args.demo:
            demo = _demo_episodes()
            if args.limit is not None:
                demo = demo[: args.limit]
            n_source = len(demo)
            def get_source(k: int) -> Episode:
                return _truncate(demo[k], args.max_frames)
            source_desc = f"{n_source} demo episodes (toy pick_place scene)"
        else:
            if not args.input:
                raise CliError("--input <lerobot-dataset> is required (or use --demo)")
            import json as _json

            import pyarrow.parquet as pq

            from lmfao.datasets.lerobot import _episode_records, _video_keys
            _info = _json.loads((Path(args.input) / "meta" / "info.json").read_text())
            _keys = _video_keys(_info)
            _on_disk = [k for k in _keys if (Path(args.input) / "videos" / k).exists()]
            if args.video_key is None and len(_on_disk) > 1:
                print(
                    f"multi-cam preserve-all: {[k.split('.')[-1] for k in _on_disk]} "
                    f"({len(_on_disk)} streams)"
                )
            if args.write_video_key is not None and args.video_key is None and len(_on_disk) > 1:
                raise CliError(
                    "--write-video-key cannot be used with multi-cam preserve-all; "
                    "pass --video-key <one> for single-stream rename, or omit "
                    "--write-video-key to keep original camera names"
                )
            recs = _episode_records(Path(args.input), pq)
            indices = [int(r["episode_index"]) for r in recs]
            if args.limit is not None:
                indices = indices[: args.limit]
            n_source = len(indices)
            def get_source(k: int) -> Episode:
                return read_lerobot_dataset(
                    args.input, video_key=args.video_key, episodes=[indices[k]],
                    max_frames=args.max_frames,
                )[0]
            source_desc = f"{n_source} episodes from {args.input}"
        if n_source == 0:
            raise CliError("no episodes loaded; nothing to do")

        # --- resume / overwrite handling ---
        writer = None
        start = 0
        if resume and ckpt.exists():
            import pickle
            try:
                saved = pickle.loads(ckpt.read_bytes())
            except Exception as e:  # noqa: BLE001
                raise CliError(f"cannot read resume checkpoint {ckpt}: {e}") from e
            if not isinstance(saved, dict) or saved.get("sig") != sig:
                raise CliError(
                    "cannot --resume: the checkpoint is unreadable or was written for a "
                    "different run (seed/config/cameras changed). Use a fresh --output or drop --resume."
                )
            first = get_source(0)
            writer = _new_writer(out, args, first)
            writer.load_state_dict(saved["writer"])
            missing = writer.missing_files()
            if missing:
                raise CliError(
                    f"cannot --resume: {len(missing)} output file(s) named in the checkpoint "
                    f"are missing on disk (e.g. {missing[0]}); the output was modified. "
                    "Use --overwrite to restart from scratch."
                )
            start = int(saved["source_done"])
            print(f"resuming: {start}/{n_source} source episodes already done")
        else:
            if (out / "meta" / "info.json").exists() and not args.overwrite:
                raise CliError(f"{out} already contains a dataset; pass --overwrite to replace it")
            _clear_dataset(out)
            first = get_source(0)
            writer = _new_writer(out, args, first)
    except (OSError, ValueError, KeyError) as e:
        raise CliError(f"cannot start augmentation: {e}") from e

    # --- stream: one source at a time -> its variants -> checkpoint ---
    generate = (
        (lambda ep, k: iter_sweep_episode(
            ep, payload, seed=args.seed, include_original=args.include_original, source_index=k))
        if mode == "sweep"
        else (lambda ep, k: iter_augment_episode(
            ep, payload, variants=args.variants, seed=args.seed,
            include_original=args.include_original,
            original_copies=getattr(args, "original_copies", 1), source_index=k))
    )
    try:
        for k in range(start, n_source):
            ep = first if k == 0 else get_source(k)
            for produced in generate(ep, k):
                writer.add_episode(produced)
            if resume:
                _checkpoint(ckpt, sig, k + 1, writer)
        writer.close()
    except (OSError, ValueError, TypeError, KeyError) as e:
        hint = "" if resume else " (re-run with --resume to make long runs crash-safe)"
        raise CliError(f"augmentation failed: {e}{hint}") from e
    if ckpt.exists():
        ckpt.unlink()

    if mode == "sweep":
        desc = f"swept {n_source} source episode(s) x {len(payload)} step(s)"
    else:
        steps = ", ".join(s.get("name", "?") for s in payload)
        desc = f"augmented {n_source} source episode(s) x {args.variants} variant(s); steps: {steps}"
    if args.include_original:
        desc += " + originals"
    print(f"loaded {source_desc}")
    print(desc)
    print(f"wrote {writer.episodes_written} episodes -> {out}")
    _print_chinchilla_guidance(n_source, writer.episodes_written)
    return 0


def _print_chinchilla_guidance(n_source: int, n_out: int) -> None:
    """Surface the training implication of growing the dataset.

    Lesson learned the hard way (stock 76% vs v2_steady 8%, Aug 2026): training
    an augmented dataset for the SAME step count as its source trains each
    sample 1/multiplier as often — the augmented policy is undertrained, and
    the comparison is confounded. With model size fixed, hold the
    chinchilla-style data/compute ratio by keeping EPOCHS (total passes over
    the data) constant: steps must scale with dataset size.
    """
    if n_source <= 0 or n_out <= 0:
        return
    mult = n_out / n_source
    print()
    print(f"chinchilla note: dataset grew {mult:.1f}x ({n_source} -> {n_out} episodes).")
    if mult > 1.001:
        print(f"  Training on this output needs ~{mult:.1f}x the STEPS of the source's")
        print("  recipe to hold epochs (passes over the data) constant — or use")
        print("  EPOCHS=<n> in train_act.sh / m1 train.sh, which derives STEPS from")
        print("  the dataset size automatically. Comparing against a source-trained")
        print(f"  policy at equal STEPS is confounded: each sample is seen 1/{mult:.1f}x as often.")


def _new_writer(out: Path, args: argparse.Namespace, first: Episode) -> LeRobotStreamingWriter:
    state_dim = first.state.shape[1] if first.state is not None else 0
    action_dim = first.actions.shape[1] if first.actions is not None else 0
    keys = first.video_keys
    if not keys:
        # Demo episodes may not stamp video_key; fall back to write/default name.
        keys = [args.write_video_key or "observation.images.augmented"]
    elif args.write_video_key is not None:
        if len(keys) > 1:
            raise CliError(
                "--write-video-key is only valid for single-camera episodes"
            )
        keys = [args.write_video_key]
        first.metadata["video_key"] = args.write_video_key
    else:
        # Ensure primary metadata matches the writer key list.
        first.metadata.setdefault("video_key", keys[0])
        first.metadata["video_keys"] = list(keys)
    return LeRobotStreamingWriter(
        out, video_keys=keys, fps=first.fps, state_dim=state_dim, action_dim=action_dim,
    )


def _checkpoint(ckpt: Path, sig: str, source_done: int, writer: LeRobotStreamingWriter) -> None:
    import pickle
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"sig": sig, "source_done": source_done, "writer": writer.state_dict()}))
    tmp.replace(ckpt)  # atomic: a crash mid-write can't corrupt the checkpoint


def _clear_dataset(out: Path) -> None:
    import shutil
    for sub in ("data", "videos", "meta"):
        p = out / sub
        if p.exists():
            shutil.rmtree(p)


def _validate_video_key(key: str) -> None:
    if (not key) or ("/" in key) or ("\\" in key) or ("\x00" in key) or key in (".", "..") \
            or key.startswith(("/", "~")):
        raise CliError(
            f"invalid --write-video-key {key!r}: must be a plain stream name "
            "(e.g. observation.images.wrist), not a path"
        )


def _truncate(ep: Episode, cap: int | None) -> Episode:
    """Truncate a demo episode to its first ``cap`` frames (real data is capped at
    read time via max_frames; the demo scene is built in memory)."""
    if cap is None or ep.num_frames <= cap:
        return ep
    return Episode(
        frames=ep.frames[:cap],
        state=None if ep.state is None else ep.state[:cap],
        actions=None if ep.actions is None else ep.actions[:cap],
        fps=ep.fps, task=ep.task,
        camera_poses=None if ep.camera_poses is None else ep.camera_poses[:cap],
        intrinsics=ep.intrinsics, metadata=ep.metadata,
        extra_videos={k: v[:cap] for k, v in ep.extra_videos.items()},
    )


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
        aug = sum(1 for p in prov if p.get("augmented") and not p.get("synthetic"))
        real = len(prov) - synth - aug
        parts = [f"{real} real"]
        if aug:
            parts.append(f"{aug} augmented")
        if synth:
            parts.append(f"{synth} synthetic")
        print(f"lmfao:     provenance sidecar present — {' / '.join(parts)} episode(s)")

    if args.episodes:
        for r in records:
            ei = int(r["episode_index"])
            task = r.get("tasks", [""])
            task = task[0] if isinstance(task, list) and task else ""
            print(f"  episode {ei}: {int(r['length'])} frames  task={task!r}")
    return 0


def _wizard(args: argparse.Namespace) -> int:
    from lmfao.wizard import run_wizard
    return run_wizard()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmfao", description="LMFAO data tools")
    # No subcommand -> the interactive wizard (see main()).
    sub = parser.add_subparsers(dest="command", required=False)

    wiz = sub.add_parser("wizard", help="interactive guided mode (default when run with no command)")
    wiz.set_defaults(func=_wizard)

    ins = sub.add_parser("inspect", help="summarize a LeRobot dataset (no video decode)")
    ins.add_argument("dataset", help="path to a LeRobot dataset root")
    ins.add_argument("--episodes", action="store_true", help="also list every episode")
    ins.set_defaults(func=_inspect)

    aug = sub.add_parser("augment", help="augment a dataset with the ADJUST pipeline (native res, no synthesis)")
    aug.add_argument("--input", help="path to a LeRobot dataset root")
    aug.add_argument("--output", required=True, help="output dataset directory")
    aug.add_argument("--config", help="pipeline JSON: a list of steps, or {\"pipeline\": [...]}")
    aug.add_argument("--demo", action="store_true", help="use the built-in toy scene instead of --input")
    aug.add_argument("--variants", type=int, default=1,
                     help="number of independently-seasoned copies per source episode (default 1)")
    aug.add_argument("--include-original", action="store_true",
                     help="also emit the un-augmented source episodes (originals + variants)")
    aug.add_argument("--original-copies", type=int, default=1,
                     help="with --include-original, emit this many copies of each original "
                          "(default 1); e.g. --original-copies 2 --variants 1 gives a "
                          "67/33 original/augmented mix")
    aug.add_argument("--seed", type=int, default=None, help="base seed")
    aug.add_argument("--video-key", default=None, help="which camera stream to read")
    aug.add_argument("--write-video-key", default=None,
                     help="video key to write (default: keep the source camera's key)")
    aug.add_argument("--limit", type=int, default=None, help="load at most N episodes")
    aug.add_argument("--max-frames", type=int, default=None, help="truncate each episode to N frames")
    aug.add_argument("--overwrite", action="store_true",
                     help="replace an existing dataset at --output instead of erroring")
    aug.add_argument("--resume", action="store_true",
                     help="checkpoint after each source episode so a killed run can continue")
    aug.set_defaults(func=_augment)

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
    func = getattr(args, "func", None)
    if func is None:
        # Bare `lmfao` with no subcommand: launch the interactive wizard.
        func = _wizard
    try:
        return func(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
