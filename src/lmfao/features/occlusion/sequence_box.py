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
    "occlusion.sequence_box",
    tags=("occlusion", "sequence", "robotics"),
    description="Apply one frame-consistent occlusion box across an observation sequence.",
)
@dataclass
class SequenceBoxOcclusion(AugmentationFeature):
    box_area_range: tuple[float, float] = (0.02, 0.20)
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    fill: FillMode = "mean"

    def __post_init__(self) -> None:
        self.box_area_range = validate_fraction_range(self.box_area_range, "box_area_range")
        self.aspect_ratio_range = validate_range(self.aspect_ratio_range, "aspect_ratio_range")
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
            "fill": self.fill,
            "temporal_mode": "sequence_constant",
        }
        output = runtime.execute(self.name, video, params, metadata)
        metadata.setdefault("augmentation_params", {})[self.name] = copy_metadata_params(params)
        return output, metadata
