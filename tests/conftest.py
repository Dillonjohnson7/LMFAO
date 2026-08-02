import numpy as np
import pytest

from lmfao.datasets import Episode
from lmfao.miniworld import default_intrinsics, look_at


def build_episode(seed: int = 0, frames: int = 5, size: int = 40, with_puck: bool = True) -> Episode:
    """A small but geometrically real episode: a camera arcs over a textured
    table plane at z=0, with a red puck blob near the centre. Camera poses are
    exact (they stand in for FK output), so reconstruction and novel-view render
    are meaningful.
    """
    h = w = size
    k = default_intrinsics(w, h, 60.0)
    imgs = np.zeros((frames, h, w, 3), np.uint8)
    poses = np.zeros((frames, 4, 4))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(frames):
        ang = -0.3 + 0.6 * i / max(frames - 1, 1)
        eye = np.array([np.sin(ang) * 0.3, -0.5 + 0.1 * i, 1.1])
        poses[i] = look_at(eye, np.array([0.0, 0.0, 0.0]))
        imgs[i, ..., 0] = (xx * 6) % 256
        imgs[i, ..., 1] = (yy * 6) % 256
        imgs[i, ..., 2] = 90
        if with_puck:
            blob = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 < 20
            imgs[i][blob] = (210, 40, 40)
    # state: 3 gripper xyz channels + 1 gripper-opening channel in the carry band
    state = np.tile(np.array([0.0, 0.0, 0.2, 30.0]), (frames, 1)).astype(float)
    return Episode(
        frames=imgs,
        state=state,
        camera_poses=poses,
        intrinsics=k,
        task="pick_place_v2",
        metadata={"episode_id": int(seed)},
    )


@pytest.fixture
def episode_factory():
    return build_episode


PUCK_COLOR = (210 / 255, 40 / 255, 40 / 255)
