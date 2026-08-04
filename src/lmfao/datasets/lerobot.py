"""LeRobot v3.0 dataset I/O — the missing keystone (plan §8a).

Reads real LeRobot datasets (video shards + parquet state) into in-memory
:class:`~lmfao.datasets.episode.Episode` objects, and writes episodes back out
in the same on-disk shape. Kept out of the numpy-only core: this module needs
``pyarrow`` (parquet) and ``av`` (video), shipped as the ``lmfao[lerobot]``
extra, so importing it without them raises a clear, actionable error.

The core LeRobot v3 layout this targets::

    meta/info.json                       fps, features, path templates
    meta/tasks.parquet                   task_index -> task string
    meta/episodes/chunk-*/file-*.parquet per-episode records (ranges, video refs)
    data/chunk-*/file-*.parquet          per-frame state/action rows
    videos/{key}/chunk-*/file-*.mp4      per-camera video shards

Camera poses are deliberately absent from LeRobot datasets (the real system
derives them from robot forward-kinematics + hand-eye calibration). Episodes
therefore come back with ``camera_poses=None``; the GENERATE path attaches poses
separately (see ``lmfao.datasets.poses``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lmfao.datasets._stats import reduce_samples, stats_from_samples, stats_to_lists
from lmfao.datasets.episode import Episode


def _require_deps() -> tuple[Any, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading/writing LeRobot datasets needs pyarrow. "
            "Install the extra: `pip install lmfao[lerobot]`."
        ) from e
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "reading/writing LeRobot videos needs PyAV. "
            "Install the extra: `pip install lmfao[lerobot]`."
        ) from e
    return pq, av


# ---------------------------------------------------------------- reading


def _video_keys(info: dict) -> list[str]:
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]


def _read_tasks(root: Path, pq) -> dict[int, str]:
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    # use_threads=False: pyarrow's threaded reader can deadlock under some
    # pyarrow/Python builds (seen with pyarrow 25 on CPython 3.14), hanging the
    # whole CLI. These files are tiny, so single-threaded reads cost nothing.
    table = pq.read_table(path, use_threads=False)
    cols = table.column_names
    # Real exports vary: some name the text column "task", others write it via a
    # pandas index that parquet preserves as "__index_level_0__". Take "task" if
    # present, else the single non-index column.
    if "task" in cols:
        text_col = "task"
    else:
        others = [c for c in cols if c != "task_index"]
        if len(others) != 1:
            return {}
        text_col = others[0]
    txt = table.column(text_col).to_pylist()
    if "task_index" in cols:
        idx = table.column("task_index").to_pylist()
    else:
        # Some exports index tasks by row position.
        idx = list(range(len(txt)))
    return {int(i): str(t) for i, t in zip(idx, txt)}


def _episode_records(root: Path, pq) -> list[dict]:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    records: list[dict] = []
    for f in files:
        table = pq.read_table(f, use_threads=False).to_pylist()
        records.extend(table)
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def _decode_frames(av, path: Path, from_ts: float, to_ts: float, expected: int) -> np.ndarray:
    """Decode RGB frames in ``[from_ts, to_ts)`` from an mp4 shard."""
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Seek a little before the target so we don't miss the first frame.
        if from_ts > 0:
            try:
                container.seek(int(max(from_ts - 0.5, 0) / stream.time_base), stream=stream)
            except Exception:
                container.seek(0)
        # Half a frame period of slack: episode boundaries come from the
        # dataset's fps while frame PTS come from the stream's rate, and the two
        # grids can differ by float rounding (e.g. 29.97 fps written as
        # 30000/1001). Anything within half a period is unambiguous.
        rate = stream.average_rate
        tol = float(1.0 / (2.0 * float(rate))) if rate else 1e-6
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if t is not None and t < from_ts - tol:
                continue
            if t is not None and t >= to_ts - tol:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            if expected and len(frames) >= expected:
                break
    if not frames:
        raise ValueError(f"no frames decoded from {path} in [{from_ts}, {to_ts})")
    return np.stack(frames, axis=0)


def read_lerobot_dataset(
    root: str | Path,
    *,
    video_key: str | None = None,
    episodes: Sequence[int] | None = None,
    limit: int | None = None,
    max_frames: int | None = None,
) -> list[Episode]:
    """Load LeRobot episodes from ``root`` into :class:`Episode` objects.

    Parameters
    ----------
    video_key:
        Which camera stream to load as the episode frames. Defaults to the first
        ``dtype == "video"`` feature in ``info.json``.
    episodes:
        Explicit episode indices to load (default: all).
    limit:
        Load at most this many episodes (applied after ``episodes``).
    max_frames:
        Truncate each episode to its first ``max_frames`` frames (state + video),
        handy for quick experiments on long clips.
    """
    pq, av = _require_deps()
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30.0))
    keys = _video_keys(info)
    if not keys:
        raise ValueError(f"no video features found in {root}/meta/info.json")
    if video_key is None:
        # Partially-downloaded datasets can declare streams whose shards were
        # never pulled; default to one that is actually on disk.
        on_disk = [k for k in keys if (root / "videos" / k).exists()]
        key = (on_disk or keys)[0]
    else:
        key = video_key
    if key not in keys:
        raise ValueError(f"video_key {key!r} not in dataset (have: {', '.join(keys)})")

    data_tpl = info["data_path"]
    video_tpl = info["video_path"]
    tasks = _read_tasks(root, pq)
    records = _episode_records(root, pq)
    # Whether the whole dataset lives in one data parquet — if so, the global
    # dataset_from/to_index range doubles as a row range within that file.
    single_data_file = (
        len({(int(r["data/chunk_index"]), int(r["data/file_index"])) for r in records}) <= 1
    )

    # Optional provenance sidecar written by write_lerobot_dataset (§8d).
    prov_path = root / "meta" / "lmfao_provenance.json"
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else None

    if episodes is not None:
        wanted = set(int(e) for e in episodes)
        records = [r for r in records if int(r["episode_index"]) in wanted]
    if limit is not None:
        if int(limit) < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        records = records[: int(limit)]

    out: list[Episode] = []
    data_cache: dict[tuple[int, int], Any] = {}
    for rec in records:
        ep_index = int(rec["episode_index"])
        length = int(rec["length"])

        # --- per-frame state / action from the data parquet ---
        dchunk = int(rec["data/chunk_index"])
        dfile = int(rec["data/file_index"])
        ckey = (dchunk, dfile)
        if ckey not in data_cache:
            dpath = root / data_tpl.format(chunk_index=dchunk, file_index=dfile)
            if not dpath.exists():
                raise ValueError(
                    f"data parquet {dpath} referenced by episode {ep_index} does not "
                    "exist; the dataset looks incomplete (partial download?)"
                )
            # Only the columns we need — the raw parquet can carry huge per-frame
            # depth/teleop columns that would be pointlessly slow to load.
            available = pq.ParquetFile(dpath).schema_arrow.names
            wanted = [
                c
                for c in ("observation.state", "action", "episode_index", "frame_index", "task_index")
                if c in available
            ]
            data_cache[ckey] = pq.read_table(dpath, columns=wanted, use_threads=False).to_pydict()
        table = data_cache[ckey]
        if "episode_index" in table:
            ep_col = np.asarray(table["episode_index"])
            rows = np.nonzero(ep_col == ep_index)[0]
        elif single_data_file and rec.get("dataset_from_index") is not None:
            rows = np.arange(int(rec["dataset_from_index"]), int(rec["dataset_to_index"]))
        else:
            raise ValueError(
                f"data parquet for episode {ep_index} has no 'episode_index' column "
                "and the episode's rows cannot be located from its record"
            )
        if "frame_index" in table:
            frame_idx = np.asarray(table["frame_index"])[rows]
            rows = rows[np.argsort(frame_idx)]

        def _col(name: str) -> np.ndarray | None:
            if name not in table:
                return None
            col = table[name]
            arr = np.asarray([col[int(i)] for i in rows], dtype=float)
            # A zero-width column is how the writer encodes "no state/actions";
            # surface it as None so absence round-trips faithfully.
            if arr.ndim == 2 and arr.shape[1] == 0:
                return None
            return arr

        state = _col("observation.state")
        actions = _col("action")

        # --- frames from the chosen video shard ---
        if f"videos/{key}/chunk_index" not in rec:
            raise ValueError(
                f"video_key {key!r} is declared in info.json but episode {ep_index}'s "
                f"record has no columns for it; pass a video_key the episodes actually "
                f"reference (have: {', '.join(k for k in keys if f'videos/{k}/chunk_index' in rec)})"
            )
        vchunk = int(rec[f"videos/{key}/chunk_index"])
        vfile = int(rec[f"videos/{key}/file_index"])
        from_ts = float(rec[f"videos/{key}/from_timestamp"])
        to_ts = float(rec[f"videos/{key}/to_timestamp"])
        vpath = root / video_tpl.format(video_key=key, chunk_index=vchunk, file_index=vfile)
        if not vpath.exists():
            on_disk = sorted(
                p.name for p in (root / "videos").glob("*") if p.is_dir()
            ) if (root / "videos").exists() else []
            raise ValueError(
                f"video shard {vpath} for episode {ep_index} (video_key {key!r}) does "
                f"not exist; streams on disk: {', '.join(on_disk) or 'none'}. "
                "Pass video_key= to pick a downloaded stream."
            )
        expected = min(length, int(max_frames)) if max_frames else length
        frames = _decode_frames(av, vpath, from_ts, to_ts, expected=expected)

        # Align lengths (decoder can hand back one extra/fewer at boundaries).
        n = frames.shape[0]
        if state is not None:
            n = min(n, state.shape[0])
        if max_frames is not None:
            n = min(n, int(max_frames))
        frames = frames[:n]
        if state is not None:
            state = state[:n]
        if actions is not None:
            actions = actions[:n]

        task = ""
        if isinstance(rec.get("tasks"), list) and rec["tasks"]:
            task = str(rec["tasks"][0])
        elif rec.get("task_index") is not None:
            task = tasks.get(int(rec["task_index"]), "")
        elif "task_index" in table and len(rows):
            task = tasks.get(int(np.asarray(table["task_index"])[rows[0]]), "")
        elif tasks:
            task = tasks.get(0, "")

        metadata: dict[str, Any] = {}
        if provenance is not None and 0 <= ep_index < len(provenance):
            # Restore stamped provenance (synthetic flag, miniworld record, ...).
            metadata.update(provenance[ep_index])
        # Reader-owned keys always describe THIS dataset, never sidecar values
        # stamped from a previous source.
        metadata["episode_index"] = ep_index
        metadata["source_dataset"] = root.name
        metadata["video_key"] = key

        out.append(
            Episode(
                frames=frames,
                state=state,
                actions=actions,
                fps=fps,
                task=task,
                camera_poses=None,
                intrinsics=None,
                metadata=metadata,
            )
        )
    return out


# ---------------------------------------------------------------- writing


def _build_info(*, video_key: str, h: int, w: int, state_dim: int, action_dim: int,
                fps: float, codec_note: str, total_episodes: int, total_frames: int,
                total_tasks: int) -> dict:
    """Build a LeRobot v3.0 info.json.

    Declares every per-frame data column as a feature (state, action, and the
    bookkeeping index/timestamp columns), matching what LeRobot's data loader
    expects: it derives the parquet schema from these features, so a data column
    that is not declared makes ``Dataset.from_parquet`` fail. Also emits
    ``splits`` and the size fields real datasets carry.
    """
    return {
        "codebase_version": "v3.0",
        "robot_type": "lmfao_synthetic",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
            "action": {"dtype": "float32", "shape": [action_dim], "names": None},
            video_key: {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "is_depth_map": False,
                    "video.height": h, "video.width": w,
                    "video.codec": codec_note, "video.pix_fmt": "yuv420p",
                    "video.fps": fps, "video.channels": 3, "has_audio": False,
                },
            },
            # Every remaining data-parquet column must be declared or the loader's
            # schema won't match the parquet.
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _write_tasks_parquet(pq, pa, path: Path, tasks: list[str]) -> None:
    """Write meta/tasks.parquet with the pandas index metadata LeRobot expects.

    LeRobot loads tasks with pandas and uses the task *string* as the DataFrame
    index; without this metadata it gets a RangeIndex and task lookups silently
    fail. (This is the ``__index_level_0__`` convention the reader also handles.)
    """
    table = pa.table({
        "task_index": pa.array(list(range(len(tasks))), type=pa.int64()),
        "task": pa.array(tasks, type=pa.string()),
    })
    pandas_meta = {
        "index_columns": ["task"],
        "column_indexes": [{
            "name": None, "field_name": None, "pandas_type": "unicode",
            "numpy_type": "object", "metadata": {"encoding": "UTF-8"},
        }],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
            {"name": "task", "field_name": "task", "pandas_type": "unicode",
             "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    table = table.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})
    pq.write_table(table, path)


def _json_default(obj: Any) -> Any:
    """JSON fallback that keeps numpy values round-trippable.

    ``default=str`` would silently turn ``np.int64(7)`` into ``"7"`` and an
    ndarray into its repr; convert to native Python containers instead and only
    stringify genuinely unserializable objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Coerce an (H, W, C) frame to contiguous uint8 RGB for video encoding."""
    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.floating):
        hi = float(arr.max()) if arr.size else 1.0
        arr = arr * (255.0 if hi <= 1.0 else 1.0)
        arr = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    c = arr.shape[-1]
    if c == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif c == 4:
        arr = arr[..., :3]
    elif c != 3:
        raise ValueError(f"unsupported channel count for video: {c}")
    return np.ascontiguousarray(arr)


def _encode_video(av, path: Path, frames: np.ndarray, fps: float, preset: str = "veryfast") -> None:
    from fractions import Fraction

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    with av.open(str(path), mode="w") as container:
        # A rational rate keeps the PTS grid on the dataset's true fps; rounding
        # to int would drift the frame times away from the fps-derived episode
        # timestamps and corrupt episode boundaries for e.g. 29.97 fps.
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1_000_000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # These are augmented training clips, not archival masters: 'veryfast'
        # encodes ~3x quicker than libx264's default 'medium' with a slightly
        # smaller file, which matters when writing many variants of 720p footage.
        if preset:
            stream.options = {"preset": preset}
        for i in range(frames.shape[0]):
            vframe = av.VideoFrame.from_ndarray(_to_uint8_rgb(frames[i]), format="rgb24")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_lerobot_dataset(
    episodes: Sequence[Episode],
    root: str | Path,
    *,
    video_key: str = "observation.images.render",
    codec_note: str = "h264",
    video_preset: str = "veryfast",
) -> Path:
    """Write episodes to ``root`` in a LeRobot v3-style layout.

    All episodes are concatenated into one video shard + one data parquet
    (``chunk-000/file-000``), which :func:`read_lerobot_dataset` round-trips.
    Every episode must share frame geometry (height/width/channels).
    """
    pq, av = _require_deps()
    import pyarrow as pa

    if not episodes:
        raise ValueError("no episodes to write")
    root = Path(root)
    ref = episodes[0]
    h, w = ref.height, ref.width
    for ep in episodes:
        if (ep.height, ep.width) != (h, w):
            raise ValueError("all episodes must share height/width to write one shard")
    if h % 2 or w % 2:
        raise ValueError(
            f"frame size {w}x{h} is not encodable: libx264/yuv420p needs even "
            "width and height. Crop or pad the frames to even dimensions first."
        )
    fps = float(ref.fps)
    for ep in episodes:
        if abs(float(ep.fps) - fps) > 1e-6:
            raise ValueError(
                f"all episodes must share fps to write one shard: got {ep.fps} and {fps}. "
                "The video is one stream, so a single frame rate applies."
            )
    if sum(ep.num_frames for ep in episodes) == 0:
        raise ValueError("refusing to write a dataset with zero frames")

    # State/actions presence must be consistent: mixing None with real arrays
    # would silently fabricate all-zero rows for the None episodes on read-back.
    has_state = {ep.state is not None for ep in episodes}
    if len(has_state) > 1:
        raise ValueError(
            "some episodes have state and others do not; write only episodes that "
            "agree on state presence so absence is not silently fabricated as zeros"
        )
    has_actions = {ep.actions is not None for ep in episodes}
    if len(has_actions) > 1:
        raise ValueError(
            "some episodes have actions and others do not; write only episodes that "
            "agree on action presence so absence is not silently fabricated as zeros"
        )

    # Replace any dataset already at this root: the writer emits a single
    # chunk-000/file-000 shard, so stale multi-shard trees left in place would
    # make _episode_records glob a hybrid of new and old records.
    for sub in ("data", "videos", "meta"):
        existing = root / sub
        if existing.exists():
            import shutil

            shutil.rmtree(existing)

    # --- concatenate frames + per-frame rows ---
    all_frames = np.concatenate([_to_uint8_rgb(ep.frames) for ep in episodes], axis=0)

    tasks: list[str] = []
    task_to_index: dict[str, int] = {}
    state_dim = next((ep.state.shape[1] for ep in episodes if ep.state is not None), 0)
    action_dim = next((ep.actions.shape[1] for ep in episodes if ep.actions is not None), 0)
    for ei, ep in enumerate(episodes):
        if ep.state is not None and ep.state.shape[1] != state_dim:
            raise ValueError(
                f"episode {ei} has state dim {ep.state.shape[1]} but the dataset "
                f"uses {state_dim}; all episodes must agree"
            )
        if ep.actions is not None and ep.actions.shape[1] != action_dim:
            raise ValueError(
                f"episode {ei} has action dim {ep.actions.shape[1]} but the dataset "
                f"uses {action_dim}; all episodes must agree"
            )

    states: list[list[float]] = []
    actions: list[list[float]] = []
    timestamps: list[float] = []
    frame_indices: list[int] = []
    episode_indices: list[int] = []
    task_indices: list[int] = []
    ep_records: list[dict] = []
    samples: dict[str, list] = {}  # per-feature reduced samples for meta/stats.json

    cursor = 0
    for ei, ep in enumerate(episodes):
        n = ep.num_frames
        if ep.task not in task_to_index:
            task_to_index[ep.task] = len(tasks)
            tasks.append(ep.task)
        tindex = task_to_index[ep.task]
        for f in range(n):
            states.append(
                list(map(float, ep.state[f])) if ep.state is not None else [0.0] * state_dim
            )
            actions.append(
                list(map(float, ep.actions[f])) if ep.actions is not None else [0.0] * action_dim
            )
            timestamps.append(f / fps)
            frame_indices.append(f)
            episode_indices.append(ei)
            task_indices.append(tindex)
        rec = {
            "episode_index": ei,
            "tasks": [ep.task],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
            f"videos/{video_key}/from_timestamp": cursor / fps,
            f"videos/{video_key}/to_timestamp": (cursor + n) / fps,
        }
        # Per-episode normalization stats (LeRobot loads the aggregate at train time).
        rgb = _to_uint8_rgb(ep.frames)  # always 3-channel; matches the written video
        img = reduce_samples(rgb, is_image=True, seed=ei)
        ep_stats = {video_key: stats_from_samples(img, count=n, image_channels=int(rgb.shape[-1]))}
        samples.setdefault(video_key, []).append(img)
        if state_dim > 0:
            sv = np.asarray(ep.state, dtype=np.float64) if ep.state is not None else np.zeros((n, state_dim))
            red = reduce_samples(sv, is_image=False)
            ep_stats["observation.state"] = stats_from_samples(red, count=n)
            samples.setdefault("observation.state", []).append(red)
        if action_dim > 0:
            av_ = np.asarray(ep.actions, dtype=np.float64) if ep.actions is not None else np.zeros((n, action_dim))
            red = reduce_samples(av_, is_image=False)
            ep_stats["action"] = stats_from_samples(red, count=n)
            samples.setdefault("action", []).append(red)
        for feat, stt in ep_stats.items():
            for sk, val in stats_to_lists(stt).items():
                rec[f"stats/{feat}/{sk}"] = val
        ep_records.append(rec)
        cursor += n

    # --- video first: encoding is the step most likely to fail, so do it before
    # any metadata lands on disk and a failure cannot leave a half-written,
    # structurally-valid-but-videoless dataset behind. ---
    _encode_video(av, root / "videos" / video_key / "chunk-000" / "file-000.mp4", all_frames, fps,
                  preset=video_preset)

    # --- info.json ---
    info = _build_info(
        video_key=video_key, h=h, w=w, state_dim=state_dim, action_dim=action_dim, fps=fps,
        codec_note=codec_note, total_episodes=len(episodes),
        total_frames=int(all_frames.shape[0]), total_tasks=len(tasks),
    )
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=1))

    # --- data parquet ---
    data_table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(list(range(len(frame_indices))), type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )
    dpath = root / "data" / "chunk-000" / "file-000.parquet"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data_table, dpath)

    # --- episodes meta parquet ---
    ep_cols: dict[str, list] = {k: [r[k] for r in ep_records] for k in ep_records[0]}
    epath = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    epath.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(ep_cols), epath)

    # --- tasks parquet (with the pandas index metadata LeRobot expects) ---
    _write_tasks_parquet(pq, pa, root / "meta" / "tasks.parquet", tasks)

    # --- aggregate stats.json (LeRobot normalizes training inputs from this) ---
    stats = {}
    for feat, reds in samples.items():
        allr = np.concatenate(reds, axis=0)
        channels = allr.shape[1] if feat == video_key else None
        st = stats_from_samples(allr, count=int(all_frames.shape[0]), image_channels=channels)
        stats[feat] = stats_to_lists(st)
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=1))

    # --- provenance sidecar (plan §8d: keep synthetic episodes distinguishable) ---
    # Reader-owned keys are recomputed on every read; persisting them here would
    # bake in stale values from the *source* dataset.
    reader_owned = {"episode_index", "source_dataset", "video_key"}
    provenance = [
        {k: v for k, v in ep.metadata.items() if k not in reader_owned} for ep in episodes
    ]
    (root / "meta" / "lmfao_provenance.json").write_text(
        json.dumps(provenance, indent=1, default=_json_default)
    )
    return root


# ---------------------------------------------------------------- streaming write


class LeRobotStreamingWriter:
    """Write a LeRobot dataset one episode at a time (bounded memory + resumable).

    Each episode is its own video + data file (``file_index`` increments), a valid
    v3 layout that :func:`read_lerobot_dataset` reads via each record's per-episode
    ranges. Unlike :func:`write_lerobot_dataset` (which concatenates every frame
    into one in-memory shard), this holds only the current episode's frames, so a
    45x14 run of 720p footage never materializes the whole dataset at once.

    ``state_dict`` / ``load_state_dict`` capture everything needed to ``close()``
    after a restart, so a caller can checkpoint after each source episode and
    ``--resume`` a killed run. All of :func:`write_lerobot_dataset`'s guards apply.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        video_key: str,
        fps: float,
        state_dim: int,
        action_dim: int,
        codec_note: str = "h264",
        video_preset: str = "veryfast",
    ) -> None:
        self._pq, self._av = _require_deps()
        self.root = Path(root)
        self.video_key = video_key
        self.fps = float(fps)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.codec_note = codec_note
        self.video_preset = video_preset
        if not (self.fps > 0):
            raise ValueError(f"fps must be positive, got {fps}")
        self._rows: list[dict] = []
        self._tasks: list[str] = []
        self._task_index: dict[str, int] = {}
        self._provenance: list[dict] = []
        self._samples: dict[str, list] = {}  # per-feature reduced samples for stats.json
        self._global_index = 0
        self._h: int | None = None
        self._w: int | None = None

    @property
    def episodes_written(self) -> int:
        return len(self._rows)

    def missing_files(self) -> list[str]:
        """Per-episode video/data files named in the current state that are not on
        disk. Used to detect a checkpoint whose output was deleted/moved before a
        --resume, so we refuse instead of finalizing a corrupt dataset."""
        missing: list[str] = []
        for ei in range(len(self._rows)):
            vpath = self.root / "videos" / self.video_key / "chunk-000" / f"file-{ei:03d}.mp4"
            dpath = self.root / "data" / "chunk-000" / f"file-{ei:03d}.parquet"
            for p in (vpath, dpath):
                if not p.exists():
                    missing.append(str(p))
        return missing

    def state_dict(self) -> dict:
        return {
            "rows": self._rows, "tasks": self._tasks, "task_index": self._task_index,
            "provenance": self._provenance, "samples": self._samples,
            "global_index": self._global_index,
            "h": self._h, "w": self._w,
            "video_key": self.video_key, "fps": self.fps,
            "state_dim": self.state_dim, "action_dim": self.action_dim,
        }

    def load_state_dict(self, state: dict) -> None:
        self._rows = state["rows"]
        self._tasks = state["tasks"]
        self._task_index = state["task_index"]
        self._provenance = state["provenance"]
        self._samples = state.get("samples", {})
        self._global_index = state["global_index"]
        self._h, self._w = state["h"], state["w"]

    def add_episode(self, ep: Episode) -> int:
        import pyarrow as pa

        frames = _to_uint8_rgb(ep.frames)
        n = int(frames.shape[0])
        if n == 0:
            raise ValueError("refusing to write a zero-frame episode")
        h, w = int(frames.shape[1]), int(frames.shape[2])
        if h % 2 or w % 2:
            raise ValueError(
                f"frame size {w}x{h} is not encodable: libx264/yuv420p needs even "
                "width and height."
            )
        if self._h is None:
            self._h, self._w = h, w
        elif (h, w) != (self._h, self._w):
            raise ValueError(
                f"episode {len(self._rows)} is {w}x{h} but the dataset is {self._w}x{self._h}; "
                "all episodes must share geometry"
            )
        if abs(float(ep.fps) - self.fps) > 1e-6:
            raise ValueError(f"episode fps {ep.fps} differs from the dataset fps {self.fps}")
        if (ep.state is not None) != (self.state_dim > 0):
            raise ValueError("episode state presence disagrees with the dataset (would fabricate zeros)")
        if ep.state is not None and ep.state.shape[1] != self.state_dim:
            raise ValueError(f"episode state dim {ep.state.shape[1]} != dataset {self.state_dim}")
        if (ep.actions is not None) != (self.action_dim > 0):
            raise ValueError("episode action presence disagrees with the dataset (would fabricate zeros)")
        if ep.actions is not None and ep.actions.shape[1] != self.action_dim:
            raise ValueError(f"episode action dim {ep.actions.shape[1]} != dataset {self.action_dim}")

        ei = len(self._rows)
        if ep.task not in self._task_index:
            self._task_index[ep.task] = len(self._tasks)
            self._tasks.append(ep.task)
        tindex = self._task_index[ep.task]

        # video first (most likely to fail); one file per episode.
        vpath = self.root / "videos" / self.video_key / "chunk-000" / f"file-{ei:03d}.mp4"
        _encode_video(self._av, vpath, frames, self.fps, preset=self.video_preset)

        states = [
            list(map(float, ep.state[f])) if ep.state is not None else [0.0] * self.state_dim
            for f in range(n)
        ]
        actions = [
            list(map(float, ep.actions[f])) if ep.actions is not None else [0.0] * self.action_dim
            for f in range(n)
        ]
        table = pa.table({
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array([f / self.fps for f in range(n)], type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([ei] * n, type=pa.int64()),
            "index": pa.array(list(range(self._global_index, self._global_index + n)), type=pa.int64()),
            "task_index": pa.array([tindex] * n, type=pa.int64()),
        })
        dpath = self.root / "data" / "chunk-000" / f"file-{ei:03d}.parquet"
        dpath.parent.mkdir(parents=True, exist_ok=True)
        self._pq.write_table(table, dpath)

        # Per-feature normalization stats (LeRobot needs these to train); also
        # accumulate the reduced samples for the aggregate meta/stats.json.
        ep_stats = self._episode_stats(frames, states, actions, n, ei)

        row = {
            "episode_index": ei,
            "tasks": [ep.task],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": ei,
            "dataset_from_index": self._global_index,
            "dataset_to_index": self._global_index + n,
            f"videos/{self.video_key}/chunk_index": 0,
            f"videos/{self.video_key}/file_index": ei,
            f"videos/{self.video_key}/from_timestamp": 0.0,
            f"videos/{self.video_key}/to_timestamp": n / self.fps,
        }
        for feat, st in ep_stats.items():
            for sk, val in stats_to_lists(st).items():
                row[f"stats/{feat}/{sk}"] = val
        self._rows.append(row)
        reader_owned = {"episode_index", "source_dataset", "video_key"}
        self._provenance.append({k: v for k, v in ep.metadata.items() if k not in reader_owned})
        self._global_index += n
        return ei

    def _episode_stats(self, frames, states, actions, n, ei) -> dict:
        ep_stats: dict[str, dict] = {}
        img = reduce_samples(frames, is_image=True, seed=ei)
        ep_stats[self.video_key] = stats_from_samples(img, count=n, image_channels=int(frames.shape[-1]))
        self._samples.setdefault(self.video_key, []).append(img)
        if self.state_dim > 0:
            red = reduce_samples(np.asarray(states, dtype=np.float64), is_image=False)
            ep_stats["observation.state"] = stats_from_samples(red, count=n)
            self._samples.setdefault("observation.state", []).append(red)
        if self.action_dim > 0:
            red = reduce_samples(np.asarray(actions, dtype=np.float64), is_image=False)
            ep_stats["action"] = stats_from_samples(red, count=n)
            self._samples.setdefault("action", []).append(red)
        return ep_stats

    def close(self) -> Path:
        import pyarrow as pa

        if not self._rows:
            raise ValueError("no episodes were written")
        info = _build_info(
            video_key=self.video_key, h=self._h, w=self._w, state_dim=self.state_dim,
            action_dim=self.action_dim, fps=self.fps, codec_note=self.codec_note,
            total_episodes=len(self._rows), total_frames=int(self._global_index),
            total_tasks=len(self._tasks),
        )
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=1))

        ep_cols = {k: [r[k] for r in self._rows] for k in self._rows[0]}
        epath = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        epath.parent.mkdir(parents=True, exist_ok=True)
        self._pq.write_table(pa.table(ep_cols), epath)
        _write_tasks_parquet(self._pq, pa, self.root / "meta" / "tasks.parquet", self._tasks)

        # Aggregate normalization stats over the whole dataset (LeRobot loads
        # these from meta/stats.json to normalize during training).
        stats = {}
        for feat, reds in self._samples.items():
            allr = np.concatenate(reds, axis=0)
            channels = allr.shape[1] if feat == self.video_key else None
            st = stats_from_samples(allr, count=int(self._global_index), image_channels=channels)
            stats[feat] = stats_to_lists(st)
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, indent=1))

        (self.root / "meta" / "lmfao_provenance.json").write_text(
            json.dumps(self._provenance, indent=1, default=_json_default)
        )
        return self.root

