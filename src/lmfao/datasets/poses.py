"""Attach camera poses to pose-less episodes for the GENERATE path.

LeRobot datasets do not carry camera poses; the real system derives them from
robot forward-kinematics + a one-time hand-eye calibration (plan §3a). Until a
proper FK backend lands, this module supplies an **assumed** camera trajectory
so the reference miniworld reconstructor/renderer can run on real footage.

This is explicitly approximate: the poses are synthesized, not measured, so the
reconstruction is a plausible-looking stand-in, not geometrically faithful to
where the camera actually was. Synthetic episodes generated this way are stamped
``metadata["miniworld"]["assumed_poses"] = True`` by the caller.
"""

from __future__ import annotations

import numpy as np

from lmfao.datasets.episode import Episode
from lmfao.miniworld.camera import default_intrinsics, look_at


def assume_camera_track(
    episode: Episode,
    *,
    radius: float = 0.35,
    height: float = 1.1,
    arc: float = 0.6,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fov_deg: float = 60.0,
) -> Episode:
    """Return a copy of ``episode`` with an assumed camera arc + intrinsics.

    The camera sweeps a shallow arc of ``arc`` radians over the scene at a fixed
    ``height``, always looking at ``target`` — the same shape as the toy demo's
    wrist-cam sweep. Existing poses/intrinsics are left untouched; only the
    missing field is synthesized, so measured calibration is never overwritten.
    """
    if episode.camera_poses is not None and episode.intrinsics is not None:
        return episode

    if episode.camera_poses is not None:
        poses = episode.camera_poses
        synthesized_poses = False
    else:
        synthesized_poses = True
        n = episode.num_frames
        poses = np.zeros((n, 4, 4))
        tgt = np.array(target, dtype=float)
        for i in range(n):
            frac = 0.0 if n == 1 else i / (n - 1)
            ang = -arc / 2 + arc * frac
            eye = np.array([np.sin(ang) * radius, -0.5 + 0.2 * frac, height])
            poses[i] = look_at(eye, tgt)

    if episode.intrinsics is not None:
        intr = episode.intrinsics
    else:
        intr = default_intrinsics(episode.width, episode.height, fov_deg)
    meta = dict(episode.metadata)
    if synthesized_poses:
        meta["assumed_poses"] = True
    return Episode(
        frames=episode.frames,
        state=episode.state,
        actions=episode.actions,
        fps=episode.fps,
        task=episode.task,
        camera_poses=poses,
        intrinsics=intr,
        metadata=meta,
    )
