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


def _encode_video(av, path: Path, frames: np.ndarray, fps: float) -> None:
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
        ep_records.append(
            {
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
        )
        cursor += n

    # --- video first: encoding is the step most likely to fail, so do it before
    # any metadata lands on disk and a failure cannot leave a half-written,
    # structurally-valid-but-videoless dataset behind. ---
    _encode_video(av, root / "videos" / video_key / "chunk-000" / "file-000.mp4", all_frames, fps)

    # --- info.json ---
    info = {
        "codebase_version": "v3.0",
        "robot_type": "lmfao_synthetic",
        "total_episodes": len(episodes),
        "total_frames": int(all_frames.shape[0]),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            video_key: {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": codec_note,
                    "video.fps": fps,
                    "video.channels": 3,
                },
            },
        },
    }
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

    # --- tasks parquet ---
    tpath = root / "meta" / "tasks.parquet"
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks))), "task": tasks}), tpath
    )

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

