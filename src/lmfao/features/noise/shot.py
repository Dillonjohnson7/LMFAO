from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.noise.accelerator import add_noise
from lmfao.registry import register_augmenter


@register_augmenter(
    "noise.shot",
    tags=("noise", "sensor", "robotics"),
    description=(
        "Adds photon shot noise, which grows with the square root of pixel "
        "intensity the way a real sensor's does under a gain bump."
    ),
)
@dataclass
class ShotNoise(Augmenter):
    """Signal-dependent grain, resampled independently for every frame.

    Gaussian noise sits at the same level everywhere. Real photon noise does
    not: it scales with the square root of the signal, so a bright tabletop
    grains up while the shadows under the arm stay comparatively clean. That is
    what raising a camera's gain in a dim room actually looks like.

    ``strength`` is the standard deviation at full white, as a fraction of the
    dtype's dynamic range. It therefore matches ``noise.gaussian``'s ``sigma``
    on a white pixel and falls off towards black.

    ``device`` selects where the random numbers are drawn. The default picks an
    accelerator when one is present, which is far faster, and falls back to the
    CPU when it is not.
    """

    strength: float = 0.05
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.strength) and self.strength >= 0):
            raise ValueError("strength must be a non-negative, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented, device = add_noise(video, "shot", self.strength, rng, self.device)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
            "device": device,
        }
        return augmented, metadata
