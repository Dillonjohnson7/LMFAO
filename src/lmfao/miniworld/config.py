from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraOffset:
    """A single novel-view offset applied to the real camera (camera frame).

    ``translation`` slides the camera in metres; ``yaw`` / ``pitch`` rotate it in
    radians. These are the "small offsets from the real cameras" the plan renders
    from.
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0

    @classmethod
    def parse(cls, value: CameraOffset | Mapping[str, Any] | Sequence[float]) -> CameraOffset:
        if isinstance(value, CameraOffset):
            return value
        if isinstance(value, Mapping):
            extra = set(value) - {"translation", "yaw", "pitch"}
            if extra:
                raise ValueError(f"unknown camera offset fields: {sorted(extra)}")
            t = tuple(float(x) for x in value.get("translation", (0.0, 0.0, 0.0)))
            if len(t) > 3:
                raise ValueError(f"camera offset translation must have at most 3 components, got {len(t)}")
            translation = t + (0.0,) * (3 - len(t))
            return cls(
                translation=translation,  # type: ignore[arg-type]
                yaw=float(value.get("yaw", 0.0)),
                pitch=float(value.get("pitch", 0.0)),
            )
        seq = [float(x) for x in value]
        translation = tuple(seq[:3]) + (0.0,) * (3 - len(seq[:3]))
        yaw = float(seq[3]) if len(seq) > 3 else 0.0
        pitch = float(seq[4]) if len(seq) > 4 else 0.0
        return cls(translation=translation, yaw=yaw, pitch=pitch)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MiniWorldConfig:
    """The GENERATE half of the combined config (plan §8f/§8g).

    Every field here manufactures new episodes; the master ``enabled`` switch
    mirrors the pipeline modules' own ``enabled`` convention, so turning it off
    degenerates the whole system to plain v1.

    Attributes
    ----------
    enabled:
        Master switch. When False the generator yields nothing.
    n_synthetic:
        How many synthetic episodes to manufacture in total.
    camera_offsets:
        Novel viewpoints to re-film from. If empty, a small default fan is used.
    object_pose_region:
        ``((min_x, min_y), (max_x, max_y))`` in-plane box for randomising the
        puck's start position (object-pose augmentation). ``None`` disables it.
    table_z, background_depth, pixel_stride, max_frames:
        Reconstruction knobs forwarded to the reference reconstructor.
    puck_color, puck_color_tol:
        Segment the puck for independent re-posing.
    seed:
        Base seed for reproducible sampling of offsets / poses.
    """

    enabled: bool = True
    n_synthetic: int = 8
    camera_offsets: tuple[CameraOffset, ...] = ()
    object_pose_region: tuple[tuple[float, float], tuple[float, float]] | None = None
    table_z: float = 0.0
    background_depth: float = 6.0
    pixel_stride: int = 2
    max_frames: int = 8
    puck_color: tuple[float, float, float] | None = None
    puck_color_tol: float = 0.25
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MiniWorldConfig:
        data = dict(data)
        offsets = tuple(CameraOffset.parse(o) for o in data.pop("camera_offsets", ()) or ())
        region = data.pop("object_pose_region", None)
        if region is not None:
            (lo, hi) = region
            region = ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))
        puck_color = data.pop("puck_color", None)
        if puck_color is not None:
            puck_color = tuple(float(c) for c in puck_color)
        seed = data.pop("seed", None)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seed must be a non-negative integer or null")
            if seed < 0:
                raise ValueError("seed must be non-negative")
        known = {
            "enabled": bool(data.pop("enabled", True)),
            "n_synthetic": int(data.pop("n_synthetic", 8)),
            "camera_offsets": offsets,
            "object_pose_region": region,
            "table_z": float(data.pop("table_z", 0.0)),
            "background_depth": float(data.pop("background_depth", 6.0)),
            "pixel_stride": int(data.pop("pixel_stride", 2)),
            "max_frames": int(data.pop("max_frames", 8)),
            "puck_color": puck_color,
            "puck_color_tol": float(data.pop("puck_color_tol", 0.25)),
            "background": tuple(data.pop("background", (0.0, 0.0, 0.0))),
            "seed": seed,
        }
        if data:
            extra = ", ".join(sorted(data))
            raise ValueError(f"unknown miniworld config fields: {extra}")
        return cls(**known)

    def default_offsets(self) -> tuple[CameraOffset, ...]:
        """A small symmetric fan of viewpoints if none were configured."""
        if self.camera_offsets:
            return self.camera_offsets
        return (
            CameraOffset(translation=(-0.06, 0.0, 0.0), yaw=-0.06),
            CameraOffset(translation=(0.06, 0.0, 0.0), yaw=0.06),
            CameraOffset(translation=(0.0, -0.04, 0.03), pitch=0.05),
        )
