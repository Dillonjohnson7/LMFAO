from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class GaussianCloud:
    """A set of 3D Gaussians: the reconstructed scene representation.

    This is a deliberately minimal stand-in for a full 3D Gaussian Splatting
    model. Each Gaussian carries a world-space mean, an isotropic scale (its
    radius in metres), an RGB colour in ``[0, 1]``, and an opacity in ``[0, 1]``.
    The real backend (diff-gaussian-rasterization / RoboSplat) swaps in
    anisotropic covariances and spherical-harmonic colour behind the same
    :class:`~lmfao.miniworld.interfaces.NovelViewRenderer` interface; nothing
    downstream of the cloud needs to change.
    """

    means: np.ndarray       # (N, 3)
    colors: np.ndarray      # (N, 3) in [0, 1]
    scales: np.ndarray      # (N,) world-space radius
    opacities: np.ndarray   # (N,) in [0, 1]

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float).reshape(-1, 3)
        n = means.shape[0]
        colors = np.asarray(self.colors, dtype=float).reshape(-1, 3)
        scales = np.asarray(self.scales, dtype=float).reshape(-1)
        opacities = np.asarray(self.opacities, dtype=float).reshape(-1)
        for label, arr, expected in (
            ("colors", colors, (n, 3)),
            ("scales", scales, (n,)),
            ("opacities", opacities, (n,)),
        ):
            if arr.shape != expected:
                raise ValueError(f"{label} must have shape {expected}, got {arr.shape}")
        self.means = means
        self.colors = np.clip(colors, 0.0, 1.0)
        self.scales = np.maximum(scales, 1e-6)
        self.opacities = np.clip(opacities, 0.0, 1.0)

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def select(self, mask: np.ndarray) -> GaussianCloud:
        mask = np.asarray(mask)
        return GaussianCloud(
            self.means[mask], self.colors[mask], self.scales[mask], self.opacities[mask]
        )

    def translated(self, offset: np.ndarray) -> GaussianCloud:
        """Rigidly shift every Gaussian's mean (used for object-pose edits)."""
        offset = np.asarray(offset, dtype=float).reshape(3)
        return replace(self, means=self.means + offset)

    def transformed(self, transform: np.ndarray) -> GaussianCloud:
        """Apply a ``4x4`` rigid transform to the means (grasp attachment, etc.)."""
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("transform must be 4x4")
        homog = np.concatenate([self.means, np.ones((len(self), 1))], axis=1)
        moved = homog @ transform.T
        return replace(self, means=moved[:, :3])

    @classmethod
    def empty(cls) -> GaussianCloud:
        return cls(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,))
        )

    @classmethod
    def concat(cls, clouds: list[GaussianCloud]) -> GaussianCloud:
        clouds = [c for c in clouds if not c.is_empty]
        if not clouds:
            return cls.empty()
        return cls(
            np.concatenate([c.means for c in clouds], axis=0),
            np.concatenate([c.colors for c in clouds], axis=0),
            np.concatenate([c.scales for c in clouds], axis=0),
            np.concatenate([c.opacities for c in clouds], axis=0),
        )


@dataclass
class Scene:
    """A reconstructed mini-world: a static splat plus a movable object.

    ``static`` holds the table, bin, and background (things that hold still, so
    the splat represents them well). ``puck`` is the segmented object whose pose
    changes over an episode; keeping it separate is what lets the generator move
    it (object-pose augmentation) or attach it to the gripper during carry. The
    arm is never in the cloud at all; it is re-rendered from kinematics.
    """

    static: GaussianCloud
    puck: GaussianCloud = field(default_factory=GaussianCloud.empty)
    puck_home: np.ndarray | None = None  # (3,) resting centre of the puck

    def render_cloud(self, puck: GaussianCloud | None = None) -> GaussianCloud:
        """Combine the static scene with a (possibly re-posed) puck for rendering."""
        puck_cloud = self.puck if puck is None else puck
        return GaussianCloud.concat([self.static, puck_cloud])

    @property
    def num_gaussians(self) -> int:
        return len(self.static) + len(self.puck)
