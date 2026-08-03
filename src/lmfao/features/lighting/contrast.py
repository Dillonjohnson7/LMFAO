from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.lighting.utils import (
    apply_pointwise,
    metadata_params,
    sample_range,
    validate_positive_range,
)
from lmfao.registry import register_augmenter


@register_augmenter(
    "lighting.contrast",
    tags=("lighting", "exposure", "contrast"),
    description=(
        "Scales pixel values around mid-gray to mimic harsher vs. flatter "
        "lighting conditions, e.g. direct sunlight's hard shadows vs. diffuse "
        "overhead light."
    ),
)
@dataclass
class ContrastScale(Augmenter):
    """Multiplicative contrast adjustment pivoted around mid-gray."""

    factor: float | None = None
    min_factor: float = 0.6
    max_factor: float = 1.6

    def __post_init__(self) -> None:
        self.min_factor, self.max_factor = validate_positive_range(
            self.min_factor,
            self.max_factor,
            "min_factor",
            "max_factor",
        )
        if self.factor is not None and not (math.isfinite(self.factor) and self.factor > 0.0):
            raise ValueError("factor must be a positive, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        factor = sample_range(self.factor, self.min_factor, self.max_factor, rng)
        pivot = _mid_gray(video)
        augmented = apply_pointwise(video, lambda x: (x - pivot) * factor + pivot)

        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(
            {
                "factor": factor,
                "pivot": pivot,
            }
        )
        return augmented, metadata


def _mid_gray(video: Video) -> float:
    if np.issubdtype(video.dtype, np.integer):
        info = np.iinfo(video.dtype)
        return (float(info.min) + float(info.max)) / 2.0
    return 0.5
