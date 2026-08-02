from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video, preserve_dtype
from lmfao.features.lighting.utils import metadata_params, sample_range, validate_positive_range
from lmfao.registry import register_augmenter


@register_augmenter(
    "lighting.brightness",
    tags=("lighting", "exposure", "brightness"),
    description=(
        "Scales overall pixel intensity to mimic exposure/ambient-light "
        "differences, e.g. a dim overhead light at night vs. a brightly lit "
        "midday scene."
    ),
)
@dataclass
class BrightnessScale(Augmenter):
    """Multiplicative exposure gain."""

    factor: float | None = None
    min_factor: float = 0.6
    max_factor: float = 1.4

    def __post_init__(self) -> None:
        self.min_factor, self.max_factor = validate_positive_range(
            self.min_factor,
            self.max_factor,
            "min_factor",
            "max_factor",
        )
        if self.factor is not None and self.factor <= 0.0:
            raise ValueError("factor must be > 0.0")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        factor = sample_range(self.factor, self.min_factor, self.max_factor, rng)
        augmented = preserve_dtype(video, video.astype(np.float32, copy=True) * factor)

        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(
            {
                "factor": factor,
            }
        )
        return augmented, metadata
