from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.datasets import Episode
from lmfao.miniworld.scene import Scene


@dataclass
class FKBeaconTracker:
    """Forward-kinematics beacons for the gripper and puck.

    Because the gripper pose is pure FK from recorded state, and the puck is
    either resting or rigidly attached to the gripper during carry, both beacon
    positions are known exactly for every frame -- so re-rendered frames come
    with free ground-truth annotations (plan §3c/§3d).

    The reference reads the gripper's world position from three configured state
    channels and reads the grasp phase from a gripper-opening channel: while that
    value sits in the "carry" band the puck is attached to the gripper (offset by
    ``grasp_offset``); otherwise the puck sits at its home pose. If the episode
    lacks the needed state, the tracker returns ``None`` and the generator simply
    omits beacons.
    """

    gripper_xyz_channels: tuple[int, int, int] | None = None
    grip_open_channel: int | None = None
    carry_low: float = 26.0
    carry_high: float = 36.0
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def gripper_world(self, episode: Episode, frame_index: int) -> np.ndarray | None:
        if episode.state is None or self.gripper_xyz_channels is None:
            return None
        ix, iy, iz = self.gripper_xyz_channels
        if max(ix, iy, iz) >= episode.state.shape[1]:
            return None
        return episode.state[frame_index, [ix, iy, iz]].astype(float)

    def is_carrying(self, episode: Episode, frame_index: int) -> bool:
        if episode.state is None or self.grip_open_channel is None:
            return False
        if self.grip_open_channel >= episode.state.shape[1]:
            return False
        value = float(episode.state[frame_index, self.grip_open_channel])
        return self.carry_low <= value <= self.carry_high

    def puck_world(self, episode: Episode, frame_index: int, scene: Scene) -> np.ndarray | None:
        if self.is_carrying(episode, frame_index):
            gripper = self.gripper_world(episode, frame_index)
            if gripper is None:
                return scene.puck_home
            return gripper + np.asarray(self.grasp_offset, dtype=float)
        return scene.puck_home
