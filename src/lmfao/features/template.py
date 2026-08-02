from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import register_augmenter


# Copy this file when starting a new feature. Do not import this module from
# `features/__init__.py`; it is a template, not an active augmentation.
@register_augmenter("example_feature")
@dataclass
class ExampleFeature(Augmenter):
    strength: float = 1.0

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = video.copy()

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return augmented, metadata
