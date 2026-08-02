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
    "occlusion.sequence_box",
    tags=("occlusion", "sequence", "robotics"),
    description="Apply one frame-consistent occlusion box across an observation sequence.",
)
@dataclass
class SequenceBoxOcclusion(Augmenter):
    box_area_range: tuple[float, float] = (0.02, 0.20)
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    fill: FillMode = "black"

    def __post_init__(self) -> None:
        self.box_area_range = validate_fraction_range(self.box_area_range, "box_area_range")
        self.aspect_ratio_range = validate_range(self.aspect_ratio_range, "aspect_ratio_range")
        self.fill = validate_fill(self.fill)

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = video.copy()
        _, height, width, _ = augmented.shape
        box = sample_box(height, width, self.box_area_range, self.aspect_ratio_range, rng)
        top = box["top"]
        left = box["left"]
        bottom = top + box["height"]
        right = left + box["width"]
        fill_region(augmented[:, top:bottom, left:right, :], self.fill)

        params = {
            "box_area_range": self.box_area_range,
            "aspect_ratio_range": self.aspect_ratio_range,
            "fill": self.fill,
            "temporal_mode": "sequence_constant",
            "box": box,
        }
        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(params)
        return augmented, metadata
