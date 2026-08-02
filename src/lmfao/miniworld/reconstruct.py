from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.datasets import Episode
from lmfao.miniworld.camera import Camera
from lmfao.miniworld.imaging import to_unit_rgb
from lmfao.miniworld.scene import GaussianCloud, Scene


@dataclass
class PointSplatReconstructor:
    """Fit a :class:`Scene` from an episode's frames and known camera poses.

    Reference strategy (numpy only, no optimisation): for a handful of frames,
    lift a strided grid of pixels into 3D by intersecting each camera ray with a
    ground plane (``table_z``), falling back to a fixed background depth for rays
    that miss it. Each lifted pixel becomes one Gaussian. This mirrors the real
    pipeline's inputs -- known FK camera poses instead of SfM -- while standing
    in for the actual splat fit, and it produces genuine parallax under a novel
    camera because points sit at plane-derived depths rather than a single plane
    distance.

    The puck is segmented by colour proximity (the reference's stand-in for
    object-Gaussian segmentation) so the generator can re-pose it independently.
    """

    table_z: float = 0.0
    background_depth: float = 6.0
    pixel_stride: int = 2
    max_frames: int = 8
    opacity: float = 0.9
    scale_factor: float = 1.4
    puck_color: tuple[float, float, float] | None = None
    puck_color_tol: float = 0.25

    def reconstruct(self, episode: Episode) -> Scene:
        if episode.camera_poses is None or episode.intrinsics is None:
            raise ValueError(
                "reconstruction needs camera_poses and intrinsics on the episode "
                "(from FK + hand-eye calibration in the real system)"
            )

        indices = self._frame_indices(episode.num_frames)
        all_means: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_scales: list[np.ndarray] = []

        for idx in indices:
            camera = Camera(episode.intrinsics, episode.camera_poses[idx], episode.width, episode.height)
            means, colors, scales = self._lift_frame(episode.frames[idx], camera)
            all_means.append(means)
            all_colors.append(colors)
            all_scales.append(scales)

        means = np.concatenate(all_means, axis=0) if all_means else np.zeros((0, 3))
        colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3))
        scales = np.concatenate(all_scales, axis=0) if all_scales else np.zeros((0,))
        opacities = np.full(means.shape[0], self.opacity)

        cloud = GaussianCloud(means, colors, scales, opacities)
        static, puck, puck_home = self._segment_puck(cloud)
        return Scene(static=static, puck=puck, puck_home=puck_home)

    def _frame_indices(self, num_frames: int) -> np.ndarray:
        count = min(self.max_frames, num_frames)
        if count <= 0:
            return np.zeros((0,), dtype=int)
        return np.unique(np.linspace(0, num_frames - 1, count).round().astype(int))

    def _lift_frame(self, frame: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = to_unit_rgb(frame)
        height, width = rgb.shape[:2]
        stride = max(1, int(self.pixel_stride))
        us = np.arange(0, width, stride)
        vs = np.arange(0, height, stride)
        grid_u, grid_v = np.meshgrid(us, vs)
        uv = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1).astype(float)

        # World-space ray directions for each pixel (camera-z = 1 sample).
        dir_world = camera.backproject(uv, np.ones(uv.shape[0])) - camera.center
        dz = dir_world[:, 2]
        center_z = camera.center[2]

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (self.table_z - center_z) / dz
        hits_plane = np.isfinite(depth) & (depth > 1e-3)
        depth = np.where(hits_plane, depth, self.background_depth)
        depth = np.clip(depth, 1e-3, self.background_depth)

        points = camera.backproject(uv, depth)
        colors = rgb[uv[:, 1].astype(int), uv[:, 0].astype(int), :]
        scales = self.scale_factor * depth * stride / camera.fx
        return points, colors, scales

    def _segment_puck(self, cloud: GaussianCloud) -> tuple[GaussianCloud, GaussianCloud, np.ndarray | None]:
        if self.puck_color is None or cloud.is_empty:
            return cloud, GaussianCloud.empty(), None
        target = np.asarray(self.puck_color, dtype=float).reshape(3)
        dist = np.linalg.norm(cloud.colors - target[None, :], axis=1)
        is_puck = dist <= self.puck_color_tol
        if not is_puck.any():
            return cloud, GaussianCloud.empty(), None
        puck = cloud.select(is_puck)
        static = cloud.select(~is_puck)
        puck_home = puck.means.mean(axis=0)
        return static, puck, puck_home
