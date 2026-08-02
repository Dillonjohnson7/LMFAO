from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.occlusion.utils import (
    FillMode,
    fill_region,
    metadata_params,
    sample_box,
    validate_fill,
    validate_fraction_range,
    validate_range,
)
from lmfao.registry import register_augmenter


@register_augmenter(
    "occlusion.moving_box",
    tags=("occlusion", "temporal", "robotics"),
    description="Move an occlusion box smoothly through an observation sequence.",
)
@dataclass
class MovingBoxOcclusion(Augmenter):
    box_area_range: tuple[float, float] = (0.03, 0.18)
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    velocity_range: tuple[float, float] = (-8.0, 8.0)
    fill: FillMode = "random_color"
    edge_bounce: bool = True

    def __post_init__(self) -> None:
        self.box_area_range = validate_fraction_range(self.box_area_range, "box_area_range")
        self.aspect_ratio_range = validate_range(self.aspect_ratio_range, "aspect_ratio_range")
        if len(self.velocity_range) != 2:
            raise ValueError("velocity_range must contain exactly two values")
        min_velocity = float(self.velocity_range[0])
        max_velocity = float(self.velocity_range[1])
        if min_velocity > max_velocity:
            raise ValueError("velocity_range minimum cannot exceed maximum")
        self.velocity_range = (min_velocity, max_velocity)
        self.fill = validate_fill(self.fill)

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = video.copy()
        frames, height, width, _ = augmented.shape
        box = sample_box(height, width, self.box_area_range, self.aspect_ratio_range, rng)
        velocity = {
            "y": float(rng.uniform(*self.velocity_range)),
            "x": float(rng.uniform(*self.velocity_range)),
        }
        positions = []

        for frame_index in range(frames):
            top = _position(box["top"], velocity["y"], frame_index, height - box["height"], self.edge_bounce)
            left = _position(box["left"], velocity["x"], frame_index, width - box["width"], self.edge_bounce)
            bottom = top + box["height"]
            right = left + box["width"]
            fill_region(augmented[frame_index, top:bottom, left:right, :], self.fill, rng)
            positions.append({"frame": frame_index, "top": top, "left": left})

        params = {
            "box_area_range": self.box_area_range,
            "aspect_ratio_range": self.aspect_ratio_range,
            "velocity_range": self.velocity_range,
            "fill": self.fill,
            "edge_bounce": self.edge_bounce,
            "temporal_mode": "smooth_motion",
            "box": {"height": box["height"], "width": box["width"]},
            "start": {"top": box["top"], "left": box["left"]},
            "velocity": velocity,
            "positions": positions,
        }
        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(params)
        return augmented, metadata


def _position(start: int, velocity: float, frame_index: int, max_position: int, edge_bounce: bool) -> int:
    if max_position <= 0:
        return 0

    raw_position = start + velocity * frame_index
    if not edge_bounce:
        return int(np.clip(round(raw_position), 0, max_position))

    period = max_position * 2
    wrapped = raw_position % period
    if wrapped > max_position:
        wrapped = period - wrapped
    return int(round(wrapped))
