from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lmfao.datasets import Episode
from lmfao.miniworld.camera import Camera
from lmfao.miniworld.scene import GaussianCloud, Scene


@runtime_checkable
class SceneReconstructor(Protocol):
    """Fits a :class:`Scene` from recorded footage.

    The numpy reference (:class:`~lmfao.miniworld.reconstruct.PointSplatReconstructor`)
    lifts frames to Gaussians using known camera poses and a planar-depth
    assumption. The production backend fits a real 3D Gaussian Splatting model
    (RoboSplat / gsplat) here instead; the return type is the contract.
    """

    def reconstruct(self, episode: Episode) -> Scene: ...


@runtime_checkable
class NovelViewRenderer(Protocol):
    """Renders a Gaussian cloud from an arbitrary camera.

    Returns the rendered frame and a boolean "hole" mask marking pixels no
    Gaussian covered (regions the reconstruction never saw), which the inpainter
    then fills.
    """

    def render(self, cloud: GaussianCloud, camera: Camera) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class BeaconTracker(Protocol):
    """Produces 3D beacon positions (gripper, puck) for a frame of an episode.

    Beacons come from forward kinematics, so every rendered frame is born with
    exact ground-truth annotations. ``frame_index`` selects the frame.
    """

    def gripper_world(self, episode: Episode, frame_index: int) -> np.ndarray | None: ...

    def puck_world(self, episode: Episode, frame_index: int, scene: Scene) -> np.ndarray | None: ...


@runtime_checkable
class Inpainter(Protocol):
    """Fills holes left by reconstruction gaps.

    ``mask`` is True where pixels need filling. The reference uses a cheap
    nearest-neighbour fill; a LaMa-class model can drop in behind this.
    """

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray: ...
