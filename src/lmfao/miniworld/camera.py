from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: intrinsics ``K`` plus a world-from-camera pose.

    The pose is a ``4x4`` rigid transform whose top-left ``3x3`` block ``R`` is
    the camera's orientation in world space and whose translation ``t`` is the
    camera centre in world space. A camera-space point ``p_c`` maps to world as
    ``p_w = R @ p_c + t``; the inverse (used for projection) is
    ``p_c = R.T @ (p_w - t)``.
    """

    intrinsics: np.ndarray  # (3, 3)
    pose: np.ndarray        # (4, 4) world-from-camera
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        k = np.asarray(self.intrinsics, dtype=float)
        pose = np.asarray(self.pose, dtype=float)
        if k.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if pose.shape != (4, 4):
            raise ValueError("pose must be 4x4")
        object.__setattr__(self, "intrinsics", k)
        object.__setattr__(self, "pose", pose)

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def center(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def fx(self) -> float:
        return float(self.intrinsics[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsics[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsics[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsics[1, 2])

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` world points into this camera's frame."""
        pts = np.asarray(points_world, dtype=float)
        return (pts - self.center) @ self.rotation

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels.

        Returns ``(uv, depth)`` where ``uv`` is ``(N, 2)`` pixel coordinates and
        ``depth`` is ``(N,)`` camera-space z. Points behind the camera keep their
        (negative) depth so callers can filter them.
        """
        cam = self.world_to_camera(points_world)
        z = cam[:, 2]
        safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.fx * cam[:, 0] / safe_z + self.cx
        v = self.fy * cam[:, 1] / safe_z + self.cy
        return np.stack([u, v], axis=1), z

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift ``(N, 2)`` pixels at ``(N,)`` depth to ``(N, 3)`` world points."""
        uv = np.asarray(uv, dtype=float)
        depth = np.asarray(depth, dtype=float)
        x = (uv[:, 0] - self.cx) / self.fx * depth
        y = (uv[:, 1] - self.cy) / self.fy * depth
        cam = np.stack([x, y, depth], axis=1)
        return cam @ self.rotation.T + self.center

    def offset(
        self,
        translation: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> Camera:
        """Return a copy of this camera nudged to a novel viewpoint.

        ``translation`` is applied in the camera's own frame (so ``+x`` slides
        right, ``+z`` dollies forward) and ``yaw`` / ``pitch`` rotate the camera
        in radians. This is exactly the "small offset from the real camera"
        motion the plan re-renders from.
        """
        t_cam = np.asarray(translation, dtype=float).reshape(3)
        r = self.rotation @ _yaw_pitch(yaw, pitch)
        center = self.center + self.rotation @ t_cam
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = center
        return Camera(self.intrinsics.copy(), pose, width=self.width, height=self.height)


def _yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return ry @ rx


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Build a world-from-camera pose looking from ``eye`` toward ``target``.

    Uses the common computer-vision convention: camera looks down ``+z``, ``x``
    points right, ``y`` points down.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target coincide")
    forward = forward / norm
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
    right = right / rn
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """A reasonable pinhole ``K`` for a given image size and horizontal FOV."""
    fov = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = fx
    return np.array(
        [[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
    )
