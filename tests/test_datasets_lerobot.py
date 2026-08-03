"""Round-trip + CLI tests for the LeRobot dataset layer.

Guarded on the [datasets] extra so the core suite still runs without pyarrow/PyAV.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

from lmfao import cli  # noqa: E402
from lmfao.datasets.lerobot_reader import LeRobotReader  # noqa: E402
from lmfao.datasets.lerobot_writer import LeRobotWriter  # noqa: E402

CAMERA = "observation.images.wrist"
H, W = 32, 48


def _features() -> dict:
    return {
        "action": {"dtype": "float32", "shape": [6], "names": None},
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        CAMERA: {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _write_synthetic(root, *, n_episodes=2, length=5, fill=50) -> list[dict]:
    """Write a tiny valid v3.0 dataset; return the source arrays for comparison."""
    writer = LeRobotWriter(root, features=_features(), fps=10.0, robot_type="test_arm")
    sources = []
    rng = np.random.default_rng(0)
    for e in range(n_episodes):
        frames = np.full((length, H, W, 3), fill + e * 10, dtype=np.uint8)
        state = rng.standard_normal((length, 6)).astype(np.float32)
        actions = rng.standard_normal((length, 6)).astype(np.float32)
        timestamps = (np.arange(length) / 10.0).astype(np.float32)
        writer.add_episode(
            {CAMERA: frames}, state=state, actions=actions, timestamps=timestamps, task="pick the cube"
        )
        sources.append({"frames": frames, "state": state, "actions": actions, "timestamps": timestamps})
    writer.close()
    return sources


def test_writer_produces_valid_v30_layout(tmp_path):
    _write_synthetic(tmp_path, n_episodes=2, length=5)

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 10
    assert info["features"][CAMERA]["info"]["video.codec"] == "h264"

    assert (tmp_path / "meta" / "tasks.parquet").exists()
    assert (tmp_path / "meta" / "stats.json").exists()
    assert (tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "videos" / CAMERA / "chunk-000" / "file-000.mp4").exists()

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 6

    # LeRobot stores count as a single-element list for EVERY feature,
    # including images (never broadcast to the feature shape).
    for feat, st in stats.items():
        assert st["count"] == [10], f"stats.json count wrong for {feat}: {st['count']}"
    assert np.asarray(stats[CAMERA]["mean"]).shape == (3, 1, 1)

    import pyarrow.parquet as pq

    ep_table = pq.read_table(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = ep_table.to_pylist()
    for row in rows:
        # Native v3.0 bookkeeping columns LeRobot's append/aggregate tooling reads.
        assert row["meta/episodes/chunk_index"] == 0
        assert row["meta/episodes/file_index"] == 0
        for feat in ("action", "observation.state", CAMERA):
            assert row[f"stats/{feat}/count"] == [5], f"episode stats count wrong for {feat}"

    # tasks.parquet must carry the pandas index metadata LeRobot's loader
    # relies on to index tasks by string.
    tasks_meta = pq.read_table(tmp_path / "meta" / "tasks.parquet").schema.metadata
    assert json.loads(tasks_meta[b"pandas"])["index_columns"] == ["task"]


def test_writer_rejects_bad_episodes(tmp_path):
    writer = LeRobotWriter(tmp_path, features=_features(), fps=10.0)
    ts = (np.arange(5) / 10.0).astype(np.float32)

    with pytest.raises(ValueError, match="frames but episode length"):
        writer.add_episode({CAMERA: np.zeros((4, H, W, 3), np.uint8)}, timestamps=ts)
    with pytest.raises(ValueError, match="non-empty"):
        writer.add_episode({CAMERA: np.zeros((0, H, W, 3), np.uint8)}, timestamps=ts[:0])
    with pytest.raises(ValueError, match="state has 3 rows"):
        writer.add_episode(
            {CAMERA: np.zeros((5, H, W, 3), np.uint8)},
            state=np.zeros((3, 6), np.float32),
            timestamps=ts,
        )
    with pytest.raises(ValueError, match="even frame dimensions"):
        writer.add_episode({CAMERA: np.zeros((5, 31, W, 3), np.uint8)}, timestamps=ts)


def test_writer_accepts_lazy_camera_providers(tmp_path):
    writer = LeRobotWriter(tmp_path, features=_features(), fps=10.0)
    calls = []

    def provider():
        calls.append(1)
        return np.full((5, H, W, 3), 90, dtype=np.uint8)

    writer.add_episode({CAMERA: provider}, timestamps=(np.arange(5) / 10.0).astype(np.float32))
    writer.close()
    assert calls == [1]
    ep = LeRobotReader(tmp_path).read_episode(0)
    assert ep.frames[CAMERA].shape == (5, H, W, 3)


def test_info_json_only_declares_written_features(tmp_path):
    writer = LeRobotWriter(tmp_path, features=_features(), fps=10.0)
    # No state/actions written -> info.json must not declare them.
    writer.add_episode(
        {CAMERA: np.full((5, H, W, 3), 90, dtype=np.uint8)},
        timestamps=(np.arange(5) / 10.0).astype(np.float32),
    )
    writer.close()
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert "observation.state" not in info["features"]
    assert "action" not in info["features"]
    assert CAMERA in info["features"]


def test_reader_rejects_truncated_video(tmp_path):
    _write_synthetic(tmp_path, n_episodes=1, length=5)
    # Corrupt the dataset the way a partial download does: fewer video frames
    # than the episode metadata claims.
    from lmfao.datasets._video import encode_mp4

    vpath = tmp_path / "videos" / CAMERA / "chunk-000" / "file-000.mp4"
    encode_mp4(vpath, np.zeros((2, H, W, 3), np.uint8), 10.0)

    reader = LeRobotReader(tmp_path)
    with pytest.raises(ValueError, match="truncated or corrupt"):
        reader.read_episode(0)


def test_reader_missing_camera_video(tmp_path):
    _write_synthetic(tmp_path, n_episodes=1, length=5)
    vpath = tmp_path / "videos" / CAMERA / "chunk-000" / "file-000.mp4"
    vpath.unlink()

    reader = LeRobotReader(tmp_path)
    assert reader.available_camera_keys() == []
    with pytest.raises(FileNotFoundError, match="declares the camera"):
        reader.read_episode(0)


def test_reader_round_trips_state_and_frames(tmp_path):
    sources = _write_synthetic(tmp_path, n_episodes=2, length=5)
    reader = LeRobotReader(tmp_path)
    assert len(reader) == 2

    # Read both episodes: episode 1 lives in its own data file and exercises the
    # global->local index offset.
    for i in range(2):
        ep = reader.read_episode(i)
        assert ep.frames[CAMERA].shape == (5, H, W, 3)
        assert ep.task == "pick the cube"
        # State/action pass through parquet losslessly.
        np.testing.assert_allclose(ep.state, sources[i]["state"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(ep.actions, sources[i]["actions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(ep.timestamps, sources[i]["timestamps"], rtol=0, atol=1e-4)


def test_cli_augments_video_but_preserves_trajectory(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_synthetic(src, n_episodes=2, length=5, fill=50)

    job = {
        "lmfao_job": 1,
        "seed": 0,
        "cameras": [CAMERA],
        "variants": [
            {
                "id": "bright",
                "suffix": "bright",
                "pipeline": [
                    {"name": "lighting.brightness", "params": {"factor": 1.8}, "probability": 1.0}
                ],
            }
        ],
    }
    summary = cli.run(src, out, job, seed=0, cameras=None, limit=None)
    assert summary == {"source_episodes": 2, "variants": 1, "written_episodes": 2}

    src_reader = LeRobotReader(src)
    out_reader = LeRobotReader(out)
    assert len(out_reader) == 2

    src_ep = src_reader.read_episode(0)
    out_ep = out_reader.read_episode(0)

    # Trajectory copied through unchanged...
    np.testing.assert_allclose(out_ep.state, src_ep.state, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out_ep.actions, src_ep.actions, rtol=0, atol=1e-6)
    # ...while the video got visibly brighter (survives the lossy re-encode).
    assert out_ep.frames[CAMERA].mean() > src_ep.frames[CAMERA].mean() + 15


def test_cli_rejects_bad_variant_before_writing(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_synthetic(src, n_episodes=1, length=5)

    job = {
        "lmfao_job": 1,
        "seed": 0,
        "variants": [
            {"id": "ok", "pipeline": [{"name": "lighting.brightness", "params": {"factor": 1.2}}]},
            {"id": "typo", "pipeline": [{"name": "lighting.brightness", "propability": 1.0}]},
        ],
    }
    with pytest.raises(SystemExit, match="variant 1"):
        cli.run(src, out, job, seed=0, cameras=None, limit=None)
    # Validation happened before any decode/encode work.
    assert not (out / "videos").exists()


def test_cli_finalizes_meta_when_a_late_episode_fails(tmp_path, monkeypatch):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_synthetic(src, n_episodes=2, length=5)

    # Episode 1's video read blows up mid-run; episode 0's output must still
    # be a loadable dataset.
    real_read_video = LeRobotReader.read_video

    def flaky_read_video(self, index, cam):
        if index == 1:
            raise RuntimeError("disk on fire")
        return real_read_video(self, index, cam)

    monkeypatch.setattr(LeRobotReader, "read_video", flaky_read_video)

    job = {
        "lmfao_job": 1,
        "seed": 0,
        "variants": [{"id": "b", "pipeline": [{"name": "lighting.brightness", "params": {"factor": 1.2}}]}],
    }
    with pytest.raises(RuntimeError, match="disk on fire"):
        cli.run(src, out, job, seed=0, cameras=None, limit=None)

    reader = LeRobotReader(out)
    assert len(reader) == 1
    assert reader.read_episode(0).frames[CAMERA].shape == (5, H, W, 3)
