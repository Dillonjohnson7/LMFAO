from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import register_augmenter


@register_augmenter(
    "feature_name",
    tags=("category",),
    description="Short human-readable description for the central hub.",
)
@dataclass
class FeatureName(Augmenter):
    strength: float = 1.0

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = video.copy()

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return augmented, metadata
