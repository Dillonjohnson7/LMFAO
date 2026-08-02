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
