from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.noise.utils import defocus_blur
from lmfao.registry import register_augmenter


@register_augmenter(
    "noise.blur",
    tags=("noise", "optics", "robotics"),
    description=(
        "Softens every frame with a Gaussian point spread, standing in for a "
        "camera that has hunted out of focus."
    ),
)
@dataclass
class DefocusBlur(Augmenter):
    """An isotropic Gaussian point spread applied to every frame.

    Cheap UVC cameras hunt for focus mid-episode, so a policy trained only on
    sharp frames can come to rely on high-frequency detail that is not always
    there. Unlike the sampled noises this is deterministic and reads a pixel's
    neighbours, so there is nothing for an accelerator to draw.

    ``radius`` is the Gaussian's standard deviation in pixels, which means it is
    tied to resolution: 2 px of blur on a 640-wide frame is 4 px on a 1280-wide
    one.
    """

    radius: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise ValueError("radius must be a positive, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = defocus_blur(video, self.radius)

        metadata.setdefault("augmentation_params", {})[self.name] = {"radius": self.radius}
        return augmented, metadata
