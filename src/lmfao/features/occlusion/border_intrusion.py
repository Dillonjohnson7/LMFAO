from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.occlusion.utils import FillMode, fill_region, metadata_params, validate_fill, validate_fraction
from lmfao.registry import register_augmenter

VALID_EDGES = ("top", "bottom", "left", "right")


@register_augmenter(
    "occlusion.border_intrusion",
    tags=("occlusion", "camera", "robotics"),
    description="Occlude a region entering from one or more image borders.",
)
@dataclass
class BorderIntrusionOcclusion(Augmenter):
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
        self.max_fraction = validate_fraction(self.max_fraction, "max_fraction")
        if self.temporal_mode not in {"constant", "per_frame"}:
            raise ValueError("temporal_mode must be either 'constant' or 'per_frame'")
        self.edges = tuple(self.edges)
        self.fill = validate_fill(self.fill)

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = video.copy()
        frames, height, width, _ = augmented.shape
        regions = []
        frame_indices = range(frames) if self.temporal_mode == "per_frame" else (None,)

        for frame_index in frame_indices:
            edge = str(rng.choice(self.edges))
            fraction = float(rng.uniform(np.finfo(float).eps, self.max_fraction))
            region = _border_region(edge, fraction, height, width)
            regions.append({"frame": frame_index, "edge": edge, "fraction": fraction, **region})
            target = augmented if frame_index is None else augmented[frame_index : frame_index + 1]
            _fill_border(target, region, self.fill)

        params = {
            "edges": self.edges,
            "max_fraction": self.max_fraction,
            "fill": self.fill,
            "temporal_mode": self.temporal_mode,
            "regions": regions,
        }
        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(params)
        return augmented, metadata


def _border_region(edge: str, fraction: float, height: int, width: int) -> dict[str, int]:
    if edge in {"top", "bottom"}:
        size = max(1, int(round(height * fraction)))
        top = 0 if edge == "top" else height - size
        return {"top": top, "left": 0, "height": size, "width": width}

    size = max(1, int(round(width * fraction)))
    left = 0 if edge == "left" else width - size
    return {"top": 0, "left": left, "height": height, "width": size}


def _fill_border(video: Video, region: dict[str, int], fill: FillMode) -> None:
    top = region["top"]
    left = region["left"]
    bottom = top + region["height"]
    right = left + region["width"]
    fill_region(video[:, top:bottom, left:right, :], fill)
