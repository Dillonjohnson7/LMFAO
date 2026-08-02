"""End-to-end demo of the parallel-switch driver.

Real episodes always flow to the pipeline. Flipping ``miniworld.enabled`` adds a
second, parallel source that manufactures synthetic episodes from a Gaussian
splat of the scene; every episode, real or synthetic, is then seasoned by the
same pixel pipeline. Run with::

    python examples/miniworld_usage.py
"""

import numpy as np

from lmfao import Episode, generate_training_set
from lmfao.miniworld import default_intrinsics, look_at


def make_episode(seed: int, frames: int = 6, size: int = 48) -> Episode:
    """A toy 'pick_place' episode: a wrist camera arcs over a table with a puck."""
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
    return Episode(frames=imgs, state=state, camera_poses=poses, intrinsics=k,
                   task="pick_place_v2", metadata={"episode_id": seed})


real_episodes = [make_episode(s) for s in range(3)]

# One combined config drives both halves: GENERATE (miniworld) + ADJUST (pipeline).
config = {
    "miniworld": {
        "enabled": True,                                   # <- the switch
        "n_synthetic": 6,
        "camera_offsets": [
            {"translation": [-0.06, 0.0, 0.0], "yaw": -0.06},
            {"translation": [0.06, 0.0, 0.0], "yaw": 0.06},
        ],
        "object_pose_region": [[-0.05, -0.05], [0.05, 0.05]],
        "puck_color": [210 / 255, 40 / 255, 40 / 255],
        "seed": 7,
    },
    "pipeline": [
        # ADJUST options: any registered augmenter composes here. spatial.random_crop
        # slots in the same way once its module lands.
        {"name": "lighting.color_temperature", "params": {"intensity": 0.35}, "probability": 1.0},
        {"name": "lighting.brightness", "params": {}, "probability": 0.5},
    ],
}

training_set = generate_training_set(real_episodes, config, seed=42)
print("switch ON :", training_set.summary())

sample = next(ep for ep in training_set.episodes if ep.is_synthetic)
print("synthetic provenance:", sample.metadata["miniworld"]["camera_offset"],
      "| inpainted:", round(sample.metadata["miniworld"]["inpainted_fraction"], 3),
      "| seasoned by:", sample.metadata.get("augmentations"))

# Flip the switch off -> byte-for-byte v1 over the real episodes only.
off = generate_training_set(real_episodes, {"miniworld": {"enabled": False}, "pipeline": config["pipeline"]}, seed=42)
print("switch OFF:", off.summary())
