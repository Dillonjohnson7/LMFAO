from __future__ import annotations

from dataclasses import dataclass

from lmfao.base import AugmentationFeature, AugmentationRuntime, Metadata, Video
from lmfao.features.occlusion.utils import (
    FillMode,
    copy_metadata_params,
    validate_fill,
    validate_fraction_range,
    validate_range,
)
from lmfao.registry import register_feature


@register_feature(
    "occlusion.moving_box",
    tags=("occlusion", "temporal", "robotics"),
    description="Move an occlusion box smoothly through an observation sequence.",
)
@dataclass
class MovingBoxOcclusion(AugmentationFeature):
    box_area_range: tuple[float, float] = (0.03, 0.18)
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    velocity_range: tuple[float, float] = (-8.0, 8.0)
    fill: FillMode = "random_color"
    edge_bounce: bool = True

    def __post_init__(self) -> None:
        self.box_area_range = validate_fraction_range(self.box_area_range, "box_area_range")
        self.aspect_ratio_range = validate_range(self.aspect_ratio_range, "aspect_ratio_range")
        self.velocity_range = validate_range(self.velocity_range, "velocity_range", positive=False)
        self.fill = validate_fill(self.fill)

    def apply(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Metadata,
    ) -> tuple[Video, Metadata]:
        params = {
            "box_area_range": self.box_area_range,
            "aspect_ratio_range": self.aspect_ratio_range,
            "velocity_range": self.velocity_range,
            "fill": self.fill,
            "edge_bounce": self.edge_bounce,
            "temporal_mode": "smooth_motion",
        }
        output = runtime.execute(self.name, video, params, metadata)
        metadata.setdefault("augmentation_params", {})[self.name] = copy_metadata_params(params)
        return output, metadata
