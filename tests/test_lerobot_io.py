import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

from lmfao.datasets import Episode, read_lerobot_dataset, write_lerobot_dataset


def _episode(task="pick", frames=6, size=32, synthetic=False):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8)
    state = np.tile([0.1, 0.2, 0.3, 30.0], (frames, 1)).astype(float)
    meta = {"episode_id": 1}
    if synthetic:
        meta["synthetic"] = True
        meta["miniworld"] = {"camera_offset": {"translation": [0.06, 0.0, 0.0]}, "inpainted_fraction": 0.05}
    return Episode(frames=imgs, state=state, actions=state.copy(), fps=30.0, task=task, metadata=meta)


def test_write_read_roundtrip(tmp_path):
    episodes = [_episode(task="pick"), _episode(task="place", synthetic=True)]
    write_lerobot_dataset(episodes, tmp_path)
    back = read_lerobot_dataset(tmp_path)

    assert len(back) == 2
    for orig, got in zip(episodes, back):
        assert got.frames.shape == orig.frames.shape
        assert got.state is not None and got.state.shape == orig.state.shape
        assert got.actions is not None
        assert got.fps == orig.fps
    # H.264 is lossy, so frames won't be identical, but geometry/dtype must hold.
    assert back[0].frames.dtype == np.uint8


def test_provenance_roundtrip(tmp_path):
    episodes = [_episode(task="pick"), _episode(task="pick", synthetic=True)]
    write_lerobot_dataset(episodes, tmp_path)
    back = read_lerobot_dataset(tmp_path)

    assert back[0].is_synthetic is False
    assert back[1].is_synthetic is True
    assert back[1].metadata["miniworld"]["camera_offset"]["translation"] == [0.06, 0.0, 0.0]


def test_limit_and_max_frames(tmp_path):
    write_lerobot_dataset([_episode(), _episode(), _episode()], tmp_path)
    back = read_lerobot_dataset(tmp_path, limit=2, max_frames=3)
    assert len(back) == 2
    assert all(e.num_frames == 3 for e in back)


def test_fractional_fps_roundtrip(tmp_path):
    """29.97 fps must not drift frames across episode boundaries."""
    episodes = []
    for k in range(3):
        frames = np.full((10, 64, 64, 3), k * 80, np.uint8)
        episodes.append(Episode(frames=frames, fps=29.97, task=f"t{k}"))
    write_lerobot_dataset(episodes, tmp_path)
    back = read_lerobot_dataset(tmp_path)
    assert [e.num_frames for e in back] == [10, 10, 10]
    # Each episode is a flat color; a leaked boundary frame would shift the mean.
    for k, ep in enumerate(back):
        assert abs(float(ep.frames.mean()) - k * 80) < 2.0


def test_odd_dimensions_rejected_before_writing(tmp_path):
    out = tmp_path / "odd"
    ep = Episode(frames=np.zeros((4, 65, 65, 3), np.uint8), fps=30.0)
    with pytest.raises(ValueError, match="even"):
        write_lerobot_dataset([ep], out)
    assert not out.exists()  # nothing partial left behind


def test_rgba_frames_roundtrip(tmp_path):
    ep = Episode(frames=np.zeros((4, 64, 48, 4), np.uint8), fps=30.0)
    write_lerobot_dataset([ep], tmp_path)
    back = read_lerobot_dataset(tmp_path)
    assert back[0].frames.shape == (4, 64, 48, 3)


def test_none_state_and_actions_roundtrip(tmp_path):
    ep = Episode(frames=np.zeros((5, 64, 64, 3), np.uint8), state=None, actions=None, fps=30.0)
    write_lerobot_dataset([ep], tmp_path)
    back = read_lerobot_dataset(tmp_path)[0]
    assert back.state is None
    assert back.actions is None


def test_ragged_state_dims_rejected(tmp_path):
    eps = [
        Episode(frames=np.zeros((4, 32, 32, 3), np.uint8), state=np.zeros((4, 7)), fps=30.0),
        Episode(frames=np.zeros((3, 32, 32, 3), np.uint8), state=np.zeros((3, 3)), fps=30.0),
    ]
    with pytest.raises(ValueError, match="state dim"):
        write_lerobot_dataset(eps, tmp_path / "rag")


def test_numpy_metadata_roundtrips_as_native_types(tmp_path):
    ep = _episode()
    ep.metadata["vec"] = np.array([1.0, 2.0, 3.0])
    ep.metadata["n"] = np.int64(7)
    write_lerobot_dataset([ep], tmp_path)
    meta = read_lerobot_dataset(tmp_path)[0].metadata
    assert meta["vec"] == [1.0, 2.0, 3.0]
    assert meta["n"] == 7 and isinstance(meta["n"], int)


def test_reader_owned_metadata_not_clobbered_by_provenance(tmp_path):
    """episode_index/source_dataset/video_key must describe the dataset being
    read, not stale values stamped from the source dataset."""
    eps = [
        Episode(
            frames=np.full((3, 32, 32, 3), i * 40, np.uint8), fps=30.0, task="t",
            metadata={"episode_index": 99, "source_dataset": "OLD", "video_key": "OLD",
                      "synthetic": i > 0},
        )
        for i in range(3)
    ]
    write_lerobot_dataset(eps, tmp_path)
    back = read_lerobot_dataset(tmp_path)
    assert [e.metadata["episode_index"] for e in back] == [0, 1, 2]
    assert all(e.metadata["source_dataset"] == tmp_path.name for e in back)
    assert [e.metadata["synthetic"] for e in back] == [False, True, True]


def test_read_without_index_columns_falls_back_to_ranges(tmp_path):
    import pyarrow.parquet as pq

    write_lerobot_dataset(
        [Episode(frames=np.full((4, 32, 32, 3), i * 60, np.uint8), fps=30.0, task=f"t{i}")
         for i in range(3)],
        tmp_path,
    )
    dpath = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(dpath)
    keep = [c for c in table.column_names if c not in ("episode_index", "frame_index")]
    pq.write_table(table.select(keep), dpath)
    back = read_lerobot_dataset(tmp_path)
    assert [e.num_frames for e in back] == [4, 4, 4]
    for i, ep in enumerate(back):
        assert abs(float(ep.frames.mean()) - i * 60) < 3.0


def test_task_resolved_via_task_index_without_embedded_tasks_list(tmp_path):
    import pyarrow.parquet as pq

    write_lerobot_dataset(
        [Episode(frames=np.zeros((4, 32, 32, 3), np.uint8), fps=30.0, task=t)
         for t in ("grab red", "grab blue", "grab red")],
        tmp_path,
    )
    epath = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(epath)
    pq.write_table(table.select([c for c in table.column_names if c != "tasks"]), epath)
    back = read_lerobot_dataset(tmp_path)
    assert [e.task for e in back] == ["grab red", "grab blue", "grab red"]


def test_read_tasks_handles_pandas_index_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lmfao.datasets.lerobot import _read_tasks

    (tmp_path / "meta").mkdir()
    pq.write_table(
        pa.table({"task_index": [0, 1], "__index_level_0__": ["teleop", "push"]}),
        tmp_path / "meta" / "tasks.parquet",
    )
    assert _read_tasks(tmp_path, pq) == {0: "teleop", 1: "push"}


def test_missing_data_parquet_gives_clear_error(tmp_path):
    write_lerobot_dataset([_episode()], tmp_path)
    (tmp_path / "data" / "chunk-000" / "file-000.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        read_lerobot_dataset(tmp_path)


def test_missing_video_shard_gives_clear_error(tmp_path):
    write_lerobot_dataset([_episode()], tmp_path)
    shard = next((tmp_path / "videos").rglob("*.mp4"))
    shard.unlink()
    with pytest.raises(ValueError, match="video shard"):
        read_lerobot_dataset(tmp_path)


def test_mixed_fps_rejected(tmp_path):
    f = np.zeros((5, 32, 32, 3), np.uint8)
    eps = [Episode(frames=f, fps=30.0), Episode(frames=f, fps=120.0)]
    with pytest.raises(ValueError, match="share fps"):
        write_lerobot_dataset(eps, tmp_path)


def test_inconsistent_state_presence_rejected(tmp_path):
    f = np.zeros((5, 32, 32, 3), np.uint8)
    st = np.arange(20.0).reshape(5, 4)
    eps = [Episode(frames=f, state=st, actions=st), Episode(frames=f)]
    with pytest.raises(ValueError, match="state"):
        write_lerobot_dataset(eps, tmp_path)


def test_zero_frame_write_rejected(tmp_path):
    ep = Episode(frames=np.zeros((0, 32, 32, 3), np.uint8), fps=30.0)
    with pytest.raises(ValueError, match="zero frames"):
        write_lerobot_dataset([ep], tmp_path)


def test_overwrite_replaces_stale_shards(tmp_path):
    # Write a 3-episode dataset, then overwrite with a 1-episode one and confirm
    # no stale episode records survive to make a hybrid dataset.
    write_lerobot_dataset([_episode(), _episode(), _episode()], tmp_path)
    write_lerobot_dataset([_episode()], tmp_path)
    back = read_lerobot_dataset(tmp_path)
    assert len(back) == 1


def test_negative_limit_rejected(tmp_path):
    write_lerobot_dataset([_episode(), _episode()], tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        read_lerobot_dataset(tmp_path, limit=-1)
