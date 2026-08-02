from __future__ import annotations

from dataclasses import dataclass

from lmfao.base import AugmentationFeature, AugmentationRuntime, Metadata, Video
from lmfao.features.occlusion.utils import FillMode, copy_metadata_params, validate_fill, validate_probability
from lmfao.registry import register_feature

VALID_EDGES = ("top", "bottom", "left", "right")


@register_feature(
    "occlusion.border_intrusion",
    tags=("occlusion", "camera", "robotics"),
    description="Occlude a region entering from one or more image borders.",
)
@dataclass
class BorderIntrusionOcclusion(AugmentationFeature):
    edges: tuple[str, ...] = ("bottom", "left", "right")
    max_fraction: float = 0.25
    fill: FillMode = "black"
    temporal_mode: str = "constant"

    def __post_init__(self) -> None:
        if not self.edges:
            raise ValueError("edges must include at least one edge")
        unknown_edges = sorted(set(self.edges) - set(VALID_EDGES))
        if unknown_edges:
            raise ValueError(f"unknown border edges: {', '.join(unknown_edges)}")
        validate_probability(self.max_fraction, "max_fraction")
        if self.max_fraction == 0.0:
            raise ValueError("max_fraction must be greater than 0")
        if self.temporal_mode not in {"constant", "per_frame"}:
            raise ValueError("temporal_mode must be either 'constant' or 'per_frame'")
        self.edges = tuple(self.edges)
        self.fill = validate_fill(self.fill)

    def apply(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Metadata,
    ) -> tuple[Video, Metadata]:
        params = {
            "edges": self.edges,
            "max_fraction": self.max_fraction,
            "fill": self.fill,
            "temporal_mode": self.temporal_mode,
        }
        output = runtime.execute(self.name, video, params, metadata)
        metadata.setdefault("augmentation_params", {})[self.name] = copy_metadata_params(params)
        return output, metadata
