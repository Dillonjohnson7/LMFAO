from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.miniworld.camera import Camera
from lmfao.miniworld.scene import GaussianCloud


@dataclass
class PointSplatRenderer:
    """A numpy reference forward renderer for a :class:`GaussianCloud`.

    This is a simplified 3D Gaussian Splatting rasteriser: it projects each
    Gaussian to a screen-space centre and radius, then alpha-composites a small
    Gaussian footprint back-to-front. It is exact enough to demonstrate novel
    views and parallax, and slow enough (pure-python per-Gaussian loop) that the
    production path uses ``diff-gaussian-rasterization`` behind the same
    :class:`~lmfao.miniworld.interfaces.NovelViewRenderer` interface.

    ``render`` returns ``(frame, hole_mask)``: an ``(H, W, 3)`` uint8 frame and
    an ``(H, W)`` boolean mask that is True where no Gaussian covered the pixel
    (a reconstruction hole to be inpainted).
    """

    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near: float = 1e-3
    max_radius_px: float = 6.0
    coverage_threshold: float = 0.35

    def render(self, cloud: GaussianCloud, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        height = int(camera.height)
        width = int(camera.width)
        if height <= 0 or width <= 0:
            raise ValueError("camera must carry positive width and height to render")

        bg = np.asarray(self.background, dtype=float).reshape(3)
        color_acc = np.zeros((height, width, 3), dtype=float)
        coverage = np.zeros((height, width), dtype=float)

        if not cloud.is_empty:
            uv, z = camera.project(cloud.means)
            in_front = z > self.near
            radius = camera.fx * cloud.scales / np.maximum(z, self.near)
            radius = np.clip(radius, 0.5, self.max_radius_px)

            idx = np.nonzero(in_front)[0]
            # Back-to-front so nearer Gaussians composite over farther ones.
            idx = idx[np.argsort(-z[idx])]

            for i in idx:
                self._splat(color_acc, coverage, uv[i], radius[i], cloud.colors[i], cloud.opacities[i])

        hole_mask = coverage < self.coverage_threshold
        frame = color_acc + bg[None, None, :] * (1.0 - coverage)[:, :, None]
        frame = np.clip(frame, 0.0, 1.0)
        return (frame * 255.0 + 0.5).astype(np.uint8), hole_mask

    def _splat(
        self,
        color_acc: np.ndarray,
        coverage: np.ndarray,
        center: np.ndarray,
        radius: float,
        color: np.ndarray,
        opacity: float,
    ) -> None:
        height, width = coverage.shape
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        x0 = max(int(np.floor(cx - r)), 0)
        x1 = min(int(np.ceil(cx + r)) + 1, width)
        y0 = max(int(np.floor(cy - r)), 0)
        y1 = min(int(np.ceil(cy + r)) + 1, height)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        gx, gy = np.meshgrid(xs, ys)
        sigma = max(r * 0.5, 0.6)
        weight = np.exp(-0.5 * ((gx - cx) ** 2 + (gy - cy) ** 2) / (sigma * sigma))
        alpha = np.clip(opacity * weight, 0.0, 1.0)  # (h, w)

        region = color_acc[y0:y1, x0:x1, :]
        keep = (1.0 - alpha)[:, :, None]
        color_acc[y0:y1, x0:x1, :] = region * keep + color[None, None, :] * alpha[:, :, None]
        cov = coverage[y0:y1, x0:x1]
        coverage[y0:y1, x0:x1] = cov * (1.0 - alpha) + alpha
