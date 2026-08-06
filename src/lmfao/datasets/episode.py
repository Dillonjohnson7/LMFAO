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
        ``(F, H, W, C)`` array of frames for the *primary* camera stream
        (``metadata["video_key"]``). This is what the pixel pipeline seasons.
    extra_videos:
        Additional camera streams keyed by LeRobot feature name
        (e.g. ``observation.images.wrist``), each ``(F, H_k, W_k, C)``. Must
        share the same frame count as ``frames``. Not stored in metadata
        (provenance is JSON).
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
        boolean ``"synthetic"`` flag. Camera names live in ``video_key`` /
        ``video_keys`` (reader-owned).
    """

    frames: np.ndarray
    state: np.ndarray | None = None
    actions: np.ndarray | None = None
    fps: float = 30.0
    task: str = ""
    camera_poses: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_videos: dict[str, np.ndarray] = field(default_factory=dict)

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

        primary = self.metadata.get("video_key")
        cleaned: dict[str, np.ndarray] = {}
        for key, arr in dict(self.extra_videos or {}).items():
            if primary is not None and key == primary:
                raise ValueError(
                    f"extra_videos must not include the primary video_key {key!r}"
                )
            a = np.asarray(arr)
            if a.ndim != 4:
                raise ValueError(f"extra_videos[{key!r}] must have shape (F, H, W, C)")
            if a.shape[-1] not in (1, 3, 4):
                raise ValueError(f"extra_videos[{key!r}] channels must be 1, 3, or 4")
            if a.shape[0] != f:
                raise ValueError(
                    f"extra_videos[{key!r}] has {a.shape[0]} frames but primary has {f}"
                )
            cleaned[str(key)] = a
        self.extra_videos = cleaned

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

    @property
    def video_keys(self) -> list[str]:
        """Ordered camera feature names: primary first, then extras (sorted)."""
        primary = self.metadata.get("video_key")
        extras = sorted(self.extra_videos)
        if primary is None:
            return extras
        return [str(primary)] + [k for k in extras if k != primary]

    def videos(self) -> dict[str, np.ndarray]:
        """All camera tensors keyed by feature name."""
        primary = self.metadata.get("video_key", "observation.images.primary")
        out = {str(primary): self.frames}
        out.update(self.extra_videos)
        return out

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
            extra_videos={k: v.copy() for k, v in self.extra_videos.items()},
        )

    def with_frames(
        self,
        frames: np.ndarray,
        *,
        extra_videos: dict[str, np.ndarray] | None = None,
        **overrides: Any,
    ) -> Episode:
        """Return a new episode with different frames, everything else preserved.

        Used by the driver to swap in pipeline-seasoned frames without disturbing
        state, poses, or provenance. Pass ``extra_videos`` to replace secondary
        streams (default: deep-copy the existing extras).
        """
        new_meta = overrides.pop("metadata", _deepcopy_metadata(self.metadata))
        if extra_videos is None:
            extras = {k: v.copy() for k, v in self.extra_videos.items()}
        else:
            extras = {k: np.asarray(v) for k, v in extra_videos.items()}
        return replace(
            self,
            frames=np.asarray(frames),
            metadata=new_meta,
            extra_videos=extras,
            **overrides,
        )


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
