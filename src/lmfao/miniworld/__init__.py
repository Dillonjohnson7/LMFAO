"""Gaussian-splat mini-world generator (the v2 plan made real).

This subsystem is the GENERATE half of LMFAO: it reconstructs a scene from real
episodes and manufactures new synthetic ones from novel viewpoints / edited
object poses. It runs as an optional, switchable episode *source* in parallel
with the real-data path -- never a stage inside the pixel pipeline (plan §1).

The package ships a complete, numpy-only reference backend so the whole flow
runs and is testable without CUDA. Every heavy piece sits behind an interface
(:mod:`lmfao.miniworld.interfaces`) so a RoboSplat / diff-gaussian-rasterization
backend can drop in unchanged:

    reconstruct (SceneReconstructor) -> render (NovelViewRenderer)
        -> inpaint (Inpainter),  annotated by beacons (BeaconTracker)

The orchestrator is :class:`MiniWorldGenerator`; the parallel-switch driver that
merges its output with real episodes and seasons everything through the pixel
pipeline is :func:`lmfao.program.generate_training_set`.
"""

from lmfao.miniworld.beacons import FKBeaconTracker
from lmfao.miniworld.camera import Camera, default_intrinsics, look_at
from lmfao.miniworld.config import CameraOffset, MiniWorldConfig
from lmfao.miniworld.generator import MiniWorldGenerator
from lmfao.miniworld.inpaint import SimpleInpainter
from lmfao.miniworld.interfaces import BeaconTracker, Inpainter, NovelViewRenderer, SceneReconstructor
from lmfao.miniworld.reconstruct import PointSplatReconstructor
from lmfao.miniworld.render import PointSplatRenderer
from lmfao.miniworld.scene import GaussianCloud, Scene

__all__ = [
    "Camera",
    "CameraOffset",
    "MiniWorldConfig",
    "MiniWorldGenerator",
    "GaussianCloud",
    "Scene",
    "PointSplatReconstructor",
    "PointSplatRenderer",
    "SimpleInpainter",
    "FKBeaconTracker",
    "SceneReconstructor",
    "NovelViewRenderer",
    "BeaconTracker",
    "Inpainter",
    "default_intrinsics",
    "look_at",
]
