"""Write a LeRobot v3.0 dataset.

Behind the ``[datasets]`` extra (pyarrow + PyAV). Emits **one episode per file**
(the ``file_index`` increments per episode); that is a valid v3.0 layout because
readers address episodes by the per-episode ranges recorded in
``meta/episodes/*.parquet``, and it sidesteps the packed-file bookkeeping the
reader still has to understand for arbitrary inputs. Packing several episodes per
file is a later size optimization, not a correctness requirement.

Video is encoded to H.264 mp4; ``state``/``actions``/``timestamps`` are written
through unchanged so an augmented dataset keeps its trajectory aligned with the
augmented frames. Per-episode and aggregated stats are populated so the output
normalizes like a natively recorded dataset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from lmfao.datasets._stats import reduce_samples, stats_from_samples, stats_to_lists
from lmfao.datasets._video import encode_mp4

_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _lazy_pa():
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the LeRobot dataset layer needs pyarrow; install it with `pip install \"lmfao[datasets]\"`"
        ) from exc
    return pa, pq


class LeRobotWriter:
    """Accumulate augmented episodes, then ``close()`` to finalize the dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        features: dict,
        fps: float,
        robot_type: str = "",
        video_codec: str = "h264",
        chunks_size: int = 1000,
        encoder: str = "libx264",
        encode_preset: str | None = "veryfast",
        encode_bitrate: int | None = None,
    ) -> None:
        self.root = Path(root)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self.features = copy.deepcopy(features)
        self.fps = float(fps)
        self.robot_type = robot_type
        self.video_codec = video_codec
        self.chunks_size = chunks_size
        # H.264 stays the on-disk/info.json codec label; `encoder` is which
        # libav encoder produces it (libx264, or a hardware encoder like
        # h264_videotoolbox).
        self.encoder = encoder
        self.encode_preset = encode_preset
        self.encode_bitrate = encode_bitrate

        self.camera_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self._rows: list[dict] = []
        self._tasks: dict[str, int] = {}
        self._global_index = 0
        self._samples: dict[str, list[np.ndarray]] = {}
        self._video_shape: dict[str, tuple[int, int, int]] = {}
        self._written_features: set[str] = set()

    def _accumulate(self, feature: str, reduced: np.ndarray) -> None:
        self._samples.setdefault(feature, []).append(reduced)

    @property
    def episodes_written(self) -> int:
        return len(self._rows)

    def state_dict(self) -> dict:
        """Serializable accumulator state for checkpoint/resume.

        Everything the writer needs to `close()` correctly after a restart. The
        stats samples are a bounded per-episode subsample (a few thousand pixels
        per feature), so this stays small even for a large dataset.
        """
        return {
            "rows": self._rows,
            "tasks": self._tasks,
            "global_index": self._global_index,
            "samples": self._samples,
            "video_shape": self._video_shape,
            "written_features": self._written_features,
        }

    def load_state_dict(self, state: dict) -> None:
        self._rows = state["rows"]
        self._tasks = state["tasks"]
        self._global_index = state["global_index"]
        self._samples = state["samples"]
        self._video_shape = state["video_shape"]
        self._written_features = state["written_features"]

    def add_episode(
        self,
        frames: dict[str, np.ndarray],
        *,
        state: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        task: str = "",
    ) -> int:
        """Write one episode.

        ``frames`` maps camera key -> ``(F, H, W, 3)`` uint8 array, or a
        zero-arg callable returning one. Callables are materialized one camera
        at a time and released after encoding, so peak memory stays bounded by
        a single camera's frames regardless of camera count.
        """
        pa, pq = _lazy_pa()
        ep = len(self._rows)

        length = len(timestamps) if timestamps is not None else None
        task_index = self._tasks.setdefault(task, len(self._tasks))

        ep_stats: dict[str, dict] = {}

        # ---- videos (one camera at a time) ----
        for cam in self.camera_keys:
            src = frames[cam]
            arr = np.asarray(src() if callable(src) else src)
            if arr.ndim != 4 or arr.shape[0] == 0:
                raise ValueError(
                    f"camera '{cam}' episode {ep}: expected non-empty (F, H, W, C) frames, got shape {arr.shape}"
                )
            if length is None:
                length = int(arr.shape[0])
            elif int(arr.shape[0]) != length:
                raise ValueError(
                    f"camera '{cam}' episode {ep}: {arr.shape[0]} frames but episode length is {length}"
                )
            self._video_shape[cam] = tuple(int(x) for x in arr.shape[1:])
            vpath = self.root / _VIDEO_PATH.format(video_key=cam, chunk_index=0, file_index=ep)
            encode_mp4(
                vpath, arr, self.fps,
                codec=self.encoder,
                preset=self.encode_preset,
                bitrate=self.encode_bitrate,
            )
            reduced = reduce_samples(arr, is_image=True, seed=ep)
            ep_stats[cam] = stats_from_samples(reduced, count=int(arr.shape[0]), image_channels=int(arr.shape[-1]))
            self._accumulate(cam, reduced)
            self._written_features.add(cam)
            del arr

        if length is None:
            raise ValueError(f"episode {ep} has no timestamps and no camera frames to infer a length from")
        if timestamps is None:
            timestamps = (np.arange(length, dtype=np.float64) / self.fps).astype(np.float32)
        for name, values in (("state", state), ("actions", actions)):
            if values is not None and len(values) != length:
                raise ValueError(
                    f"episode {ep}: {name} has {len(values)} rows but episode length is {length}"
                )

        # ---- per-frame data ----
        per_frame: dict[str, np.ndarray] = {
            "timestamp": np.asarray(timestamps, dtype=np.float32).reshape(-1),
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep, dtype=np.int64),
            "index": np.arange(self._global_index, self._global_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
        }
        if state is not None:
            per_frame["observation.state"] = np.asarray(state, dtype=np.float32)
        if actions is not None:
            per_frame["action"] = np.asarray(actions, dtype=np.float32)

        table = self._build_data_table(pa, per_frame)
        dpath = self.root / _DATA_PATH.format(chunk_index=0, file_index=ep)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dpath)

        for col, values in per_frame.items():
            reduced = reduce_samples(values, is_image=False)
            ep_stats[col] = stats_from_samples(reduced, count=length)
            self._accumulate(col, reduced)
            self._written_features.add(col)

        # ---- episode metadata row ----
        row: dict = {
            "episode_index": ep,
            "tasks": [task],
            "length": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": 0,
            "data/file_index": ep,
            "dataset_from_index": self._global_index,
            "dataset_to_index": self._global_index + length,
        }
        for cam in self.camera_keys:
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = ep
            row[f"videos/{cam}/from_timestamp"] = 0.0
            row[f"videos/{cam}/to_timestamp"] = length / self.fps
        for feat, st in ep_stats.items():
            for sk, val in stats_to_lists(st).items():
                row[f"stats/{feat}/{sk}"] = val

        self._rows.append(row)
        self._global_index += length
        return ep

    @staticmethod
    def _build_data_table(pa, per_frame: dict[str, np.ndarray]):
        arrays: dict = {}
        for col, values in per_frame.items():
            v = np.asarray(values)
            if v.ndim == 2:  # vector feature -> list<float32>
                arrays[col] = pa.array(v.astype(np.float32).tolist(), type=pa.list_(pa.float32()))
            elif v.dtype.kind == "f":
                arrays[col] = pa.array(v.reshape(-1), type=pa.float32())
            else:
                arrays[col] = pa.array(v.reshape(-1), type=pa.int64())
        return pa.table(arrays)

    def close(self) -> None:
        pa, pq = _lazy_pa()
        total_episodes = len(self._rows)
        total_frames = self._global_index

        (self.root / "meta" / "info.json").write_text(
            json.dumps(self._build_info(total_episodes, total_frames), indent=4)
        )

        tasks_sorted = sorted(self._tasks.items(), key=lambda kv: kv[1])
        tasks_table = pa.table(
            {
                "task_index": pa.array([i for _, i in tasks_sorted], type=pa.int64()),
                "task": pa.array([t for t, _ in tasks_sorted], type=pa.string()),
            }
        )
        # LeRobot loads tasks with pandas and expects the task strings as the
        # DataFrame index; without this metadata it gets a RangeIndex and task
        # lookups silently fail.
        pandas_meta = {
            "index_columns": ["task"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": "task",
                    "field_name": "task",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.0.0",
        }
        tasks_table = tasks_table.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})
        pq.write_table(tasks_table, self.root / "meta" / "tasks.parquet")

        ep_dir = self.root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self._rows), ep_dir / "file-000.parquet")

        stats = {}
        for feat, reds in self._samples.items():
            all_reduced = np.concatenate(reds, axis=0)
            is_image = feat in self.camera_keys
            channels = all_reduced.shape[1] if is_image else None
            st = stats_from_samples(all_reduced, count=total_frames, image_channels=channels)
            stats[feat] = stats_to_lists(st)
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    def _build_info(self, total_episodes: int, total_frames: int) -> dict:
        features = copy.deepcopy(self.features)
        if self._written_features:
            # Only declare features that were actually written; declaring e.g.
            # observation.state when the data parquet lacks it breaks loaders.
            features = {k: v for k, v in features.items() if k in self._written_features}
        for cam in self.camera_keys:
            if cam not in features:
                continue
            h, w, c = self._video_shape.get(cam, tuple(features[cam].get("shape", (0, 0, 3))))
            features[cam]["shape"] = [h, w, c]
            features[cam]["info"] = {
                "is_depth_map": False,
                "video.height": h,
                "video.width": w,
                "video.codec": self.video_codec,
                "video.pix_fmt": "yuv420p",
                "video.fps": self.fps,
                "video.channels": c,
                "has_audio": False,
            }
        return {
            "codebase_version": "v3.0",
            "fps": self.fps,
            "robot_type": self.robot_type,
            "features": features,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "data_path": _DATA_PATH,
            "video_path": _VIDEO_PATH,
            "splits": {"train": f"0:{total_episodes}"},
        }
