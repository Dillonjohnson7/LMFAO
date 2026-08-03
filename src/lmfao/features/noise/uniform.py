from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.noise.accelerator import add_noise
from lmfao.registry import register_augmenter


@register_augmenter(
    "noise.uniform",
    tags=("noise", "quantisation", "robotics"),
    description=(
        "Adds zero-mean uniform noise, resampled independently for every frame, "
        "as a stand-in for quantisation artefacts."
    ),
)
@dataclass
class UniformNoise(Augmenter):
    """Noise drawn evenly from a band centred on zero.

    Flatter and more clipped-looking than Gaussian grain, which makes it a
    reasonable stand-in for quantisation artefacts.

    ``amplitude`` is the peak-to-peak width of the band as a fraction of the
    dtype's dynamic range.

    ``device`` selects where the random numbers are drawn. The default picks an
    accelerator when one is present, which is far faster, and falls back to the
    CPU when it is not.
    """

    amplitude: float = 0.05
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.amplitude) and self.amplitude >= 0):
            raise ValueError("amplitude must be a non-negative, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented, device = add_noise(video, "uniform", self.amplitude, rng, self.device)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "amplitude": self.amplitude,
            "device": device,
        }
        return augmented, metadata
