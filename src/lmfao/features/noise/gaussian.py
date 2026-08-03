from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.noise.accelerator import add_noise
from lmfao.registry import register_augmenter


@register_augmenter(
    "noise.gaussian",
    tags=("noise", "sensor", "robotics"),
    description=(
        "Adds zero-mean Gaussian sensor noise, resampled independently for "
        "every frame to mimic the grain of a real camera."
    ),
)
@dataclass
class GaussianNoise(Augmenter):
    """Zero-mean Gaussian noise, resampled independently for every frame.

    Independent per-frame sampling is what real sensor noise looks like: the
    grain flickers between frames instead of sitting still like a dirty lens.

    ``sigma`` is the standard deviation as a fraction of the dtype's dynamic
    range, so ``sigma=0.05`` means 5% noise whether the video is uint8 or float.

    ``device`` selects where the random numbers are drawn. The default picks an
    accelerator when one is present, which is far faster, and falls back to the
    CPU when it is not.
    """

    sigma: float = 0.05
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.sigma) and self.sigma >= 0):
            raise ValueError("sigma must be a non-negative, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented, device = add_noise(video, "gaussian", self.sigma, rng, self.device)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "sigma": self.sigma,
            "device": device,
        }
        return augmented, metadata
