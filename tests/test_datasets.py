import numpy as np
import pytest

from lmfao.datasets import Episode, stack_frames


def _frames(f=3, h=8, w=8, c=3):
    return np.zeros((f, h, w, c), np.uint8)


def test_episode_requires_4d_frames():
    with pytest.raises(ValueError):
        Episode(frames=np.zeros((8, 8, 3), np.uint8))


def test_episode_state_row_mismatch_raises():
    with pytest.raises(ValueError):
        Episode(frames=_frames(f=3), state=np.zeros((2, 4)))


def test_episode_camera_pose_shape_validated():
    with pytest.raises(ValueError):
        Episode(frames=_frames(f=3), camera_poses=np.zeros((3, 3, 3)))


def test_episode_geometry_properties():
    ep = Episode(frames=_frames(f=5, h=12, w=16, c=3))
    assert (ep.num_frames, ep.height, ep.width, ep.channels) == (5, 12, 16, 3)
    assert ep.is_synthetic is False


def test_copy_is_independent():
    ep = Episode(frames=_frames(), state=np.ones((3, 2)), metadata={"a": {"b": 1}})
    clone = ep.copy()
    clone.frames[0, 0, 0, 0] = 255
    clone.state[0, 0] = 9
    clone.metadata["a"]["b"] = 2
    assert ep.frames[0, 0, 0, 0] == 0
    assert ep.state[0, 0] == 1
    assert ep.metadata["a"]["b"] == 1


def test_with_frames_preserves_state_and_metadata():
    ep = Episode(frames=_frames(), state=np.ones((3, 2)), task="t", metadata={"synthetic": True})
    new = ep.with_frames(np.full((3, 8, 8, 3), 5, np.uint8))
    assert new.frames[0, 0, 0, 0] == 5
    np.testing.assert_array_equal(new.state, ep.state)
    assert new.task == "t"
    assert new.is_synthetic is True
    # original untouched
    assert ep.frames[0, 0, 0, 0] == 0


def test_stack_frames_concatenates():
    eps = [Episode(frames=_frames(f=2)), Episode(frames=_frames(f=3))]
    stacked = stack_frames(eps)
    assert stacked.shape == (5, 8, 8, 3)


def test_stack_frames_rejects_mismatched_geometry():
    eps = [Episode(frames=_frames(w=8)), Episode(frames=_frames(w=16))]
    with pytest.raises(ValueError):
        stack_frames(eps)
