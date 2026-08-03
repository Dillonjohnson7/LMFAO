from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass
class Episode:
    """One demonstration: a clip of frames plus the robot state that produced it.

    This is the shared currency between the v1 augmentation pipeline and the v2
    miniworld generator. A real episode is loaded from a LeRobot dataset; a
    synthetic one is manufactured by :mod:`lmfao.miniworld`. Both look identical
    to everything downstream, and provenance lives only in ``metadata`` so the
    pipeline stays source-blind.

    Attributes
    ----------
    frames:
        ``(F, H, W, C)`` array of frames, matching the ``Augmenter`` contract.
    state:
        Optional ``(F, D)`` per-frame robot state (joint angles, gripper, ...).
    actions:
        Optional ``(F, A)`` per-frame action targets.
    fps:
        Playback / recording rate, carried through so re-rendered episodes keep
        their timing.
    task:
        Free-text task label (e.g. ``"pick_place_v2"``).
    camera_poses:
        Optional ``(F, 4, 4)`` world-from-camera matrices. Required by miniworld
        (it re-renders from novel poses); unused by the pixel pipeline. In the
        real system these come from forward kinematics + hand-eye calibration.
    intrinsics:
        Optional ``(3, 3)`` pinhole camera matrix shared across frames.
    metadata:
        Free-form dict. Provenance is stamped here under ``"miniworld"`` and a
        boolean ``"synthetic"`` flag.
    """

    frames: np.ndarray
    state: np.ndarray | None = None
    actions: np.ndarray | None = None
    fps: float = 30.0
    task: str = ""
    camera_poses: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim != 4:
            raise ValueError("frames must have shape (frames, height, width, channels)")
        if frames.shape[-1] not in (1, 3, 4):
            raise ValueError("frames channels must be grayscale, RGB, or RGBA")
        self.frames = frames

        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps must be a positive, finite number, got {self.fps}")

        f = frames.shape[0]
        if self.state is not None:
            self.state = self._check_per_frame(self.state, f, "state")
        if self.actions is not None:
            self.actions = self._check_per_frame(self.actions, f, "actions")
        if self.camera_poses is not None:
            poses = np.asarray(self.camera_poses, dtype=float)
            if poses.shape != (f, 4, 4):
                raise ValueError(f"camera_poses must have shape ({f}, 4, 4), got {poses.shape}")
            self.camera_poses = poses
        if self.intrinsics is not None:
            k = np.asarray(self.intrinsics, dtype=float)
            if k.shape != (3, 3):
                raise ValueError(f"intrinsics must have shape (3, 3), got {k.shape}")
            self.intrinsics = k

    @staticmethod
    def _check_per_frame(array: np.ndarray, num_frames: int, label: str) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"{label} must be 2D (frames, dims)")
        if arr.shape[0] != num_frames:
            raise ValueError(f"{label} must have {num_frames} rows to match frames, got {arr.shape[0]}")
        return arr

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])

    @property
    def channels(self) -> int:
        return int(self.frames.shape[3])

    @property
    def is_synthetic(self) -> bool:
        return bool(self.metadata.get("synthetic", False))

    def copy(self) -> Episode:
        """Deep-ish copy: arrays and metadata are duplicated, so edits are safe."""
        return replace(
            self,
            frames=self.frames.copy(),
            state=None if self.state is None else self.state.copy(),
            actions=None if self.actions is None else self.actions.copy(),
            camera_poses=None if self.camera_poses is None else self.camera_poses.copy(),
            intrinsics=None if self.intrinsics is None else self.intrinsics.copy(),
            metadata=_deepcopy_metadata(self.metadata),
        )

    def with_frames(self, frames: np.ndarray, **overrides: Any) -> Episode:
        """Return a new episode with different frames, everything else preserved.

        Used by the driver to swap in pipeline-seasoned frames without disturbing
        state, poses, or provenance.
        """
        new_meta = overrides.pop("metadata", _deepcopy_metadata(self.metadata))
        return replace(self, frames=np.asarray(frames), metadata=new_meta, **overrides)


def _deepcopy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(dict(metadata))


def stack_frames(episodes: list[Episode]) -> np.ndarray:
    """Concatenate several episodes' frames along the frame axis.

    Convenience for callers that want a single training tensor. Raises if frame
    geometry is inconsistent.
    """
    if not episodes:
        raise ValueError("no episodes to stack")
    shapes = {ep.frames.shape[1:] for ep in episodes}
    if len(shapes) != 1:
        raise ValueError(f"episodes have mismatched frame geometry: {sorted(shapes)}")
    return np.concatenate([ep.frames for ep in episodes], axis=0)
