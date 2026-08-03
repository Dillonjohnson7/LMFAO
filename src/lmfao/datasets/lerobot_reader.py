"""Read a LeRobot v3.0 dataset into memory.

Behind the ``[datasets]`` extra (pyarrow + PyAV). Understands the v3.0 layout
where several episodes are packed into shared data-parquet and video files: each
episode's per-frame rows are the ``[dataset_from_index, dataset_to_index)`` slice
of its data file, and its frames are the ``[from_timestamp, to_timestamp)`` window
of its (possibly shared) video file. Pixel augmentation never touches the robot
trajectory, so ``state``/``actions``/``timestamps`` are read verbatim and carried
straight through to the writer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lmfao.datasets._video import decode_window


@dataclass
class LoadedEpisode:
    index: int
    frames: dict[str, np.ndarray]  # camera key -> (F, H, W, 3) uint8
    state: np.ndarray | None
    actions: np.ndarray | None
    timestamps: np.ndarray | None
    fps: float
    task: str
    task_index: int
    length: int


def _lazy_pq():
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the LeRobot dataset layer needs pyarrow; install it with `pip install \"lmfao[datasets]\"`"
        ) from exc
    return pq


class LeRobotReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        self.features: dict = self.info["features"]
        self.camera_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self.data_path_tmpl = self.info["data_path"]
        self.video_path_tmpl = self.info["video_path"]

        pq = _lazy_pq()
        tasks = pq.read_table(self.root / "meta" / "tasks.parquet").to_pydict()
        self.tasks = {int(i): t for i, t in zip(tasks["task_index"], tasks["task"])}
        self._episodes = self._load_episode_rows(pq)
        # A data file holds a contiguous range of GLOBAL row indices, but starts at
        # local row 0. Track each file's base global index so per-episode slices use
        # local offsets (works for both one-episode-per-file and packed layouts).
        self._data_file_base: dict[tuple[int, int], int] = {}
        for r in self._episodes:
            key = (r["data/chunk_index"], r["data/file_index"])
            base = self._data_file_base.get(key)
            if base is None or r["dataset_from_index"] < base:
                self._data_file_base[key] = int(r["dataset_from_index"])

    def _load_episode_rows(self, pq) -> list[dict]:
        rows: list[dict] = []
        for f in sorted((self.root / "meta" / "episodes").rglob("*.parquet")):
            rows.extend(pq.read_table(f).to_pylist())
        rows.sort(key=lambda r: r["episode_index"])
        return rows

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames", 0))

    def available_camera_keys(self, index: int = 0) -> list[str]:
        """Declared cameras whose video file for ``index`` exists on disk.

        Real datasets sometimes declare cameras whose videos were never
        uploaded; callers that default to "all cameras" should use this.
        """
        if not self._episodes:
            return []
        ep = self._episodes[index]
        return [cam for cam in self.camera_keys if self._video_path(ep, cam).exists()]

    def _video_path(self, ep: dict, cam: str) -> Path:
        return self.root / self.video_path_tmpl.format(
            video_key=cam,
            chunk_index=ep[f"videos/{cam}/chunk_index"],
            file_index=ep[f"videos/{cam}/file_index"],
        )

    def _read_data_table(self, pq, data_path: Path):
        # Episodes are usually packed several to a file; cache the last table so
        # sequential reads don't re-parse the same parquet per episode.
        cached = getattr(self, "_data_cache", None)
        if cached is not None and cached[0] == data_path:
            return cached[1]
        table = pq.read_table(data_path)
        self._data_cache = (data_path, table)
        return table

    def read_video(self, index: int, cam: str) -> np.ndarray:
        """Decode one camera of one episode, verifying the frame count."""
        ep = self._episodes[index]
        length = int(ep["length"])
        path = self._video_path(ep, cam)
        if not path.exists():
            raise FileNotFoundError(
                f"episode {index} camera '{cam}': video file {path} does not exist "
                "(this dataset declares the camera but ships no video for it)"
            )
        t0 = float(ep[f"videos/{cam}/from_timestamp"])
        t1 = float(ep[f"videos/{cam}/to_timestamp"])
        frames = decode_window(path, t0, t1, expected=length)
        if frames.shape[0] != length:
            raise ValueError(
                f"episode {index} camera '{cam}': decoded {frames.shape[0]} frames from {path} "
                f"but the episode metadata says {length}; the video is truncated or corrupt"
            )
        return frames

    def read_episode(self, index: int, cameras: list[str] | None = None) -> LoadedEpisode:
        ep = self._episodes[index]
        cams = self.camera_keys if cameras is None else list(cameras)
        length = int(ep["length"])

        state = actions = timestamps = None
        task_index = 0
        data_file = self.data_path_tmpl.format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        data_path = self.root / data_file
        pq = _lazy_pq()
        if data_path.exists():
            base = self._data_file_base[(ep["data/chunk_index"], ep["data/file_index"])]
            lo = int(ep["dataset_from_index"]) - base
            hi = int(ep["dataset_to_index"]) - base
            sub = self._read_data_table(pq, data_path).slice(lo, hi - lo).to_pydict()
            if "observation.state" in sub:
                state = np.asarray(sub["observation.state"], dtype=np.float32)
            if "action" in sub:
                actions = np.asarray(sub["action"], dtype=np.float32)
            if "timestamp" in sub:
                timestamps = np.asarray(sub["timestamp"], dtype=np.float32).reshape(-1)
            if "task_index" in sub and sub["task_index"]:
                task_index = int(sub["task_index"][0])

        frames: dict[str, np.ndarray] = {cam: self.read_video(index, cam) for cam in cams}

        task = ep["tasks"][0] if ep.get("tasks") else self.tasks.get(task_index, "")
        return LoadedEpisode(
            index=index,
            frames=frames,
            state=state,
            actions=actions,
            timestamps=timestamps,
            fps=self.fps,
            task=task,
            task_index=task_index,
            length=length,
        )
