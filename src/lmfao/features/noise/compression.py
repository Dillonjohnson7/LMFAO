from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.noise.utils import jpeg_artifacts
from lmfao.registry import register_augmenter


@register_augmenter(
    "noise.compression",
    tags=("noise", "codec", "robotics"),
    description=(
        "Round-trips every frame through JPEG's 8x8 block quantisation, the "
        "blocking and ringing a real MJPEG camera feed produces."
    ),
)
@dataclass
class CompressionArtifacts(Augmenter):
    """Blocking and ringing left behind by lossy codec quantisation.

    Webcams stream MJPEG and datasets are stored as encoded video, so this is a
    degradation the real capture path introduces rather than one we are
    inventing. It is also structured where the sampled noises are not: the error
    lands on 8x8 block edges and around sharp features, which is nothing like
    the even grain ``noise.gaussian`` adds, and ``noise.uniform`` only gestures
    at it.

    ``quality`` is the familiar JPEG scale, where 100 is near-lossless and 20 is
    visibly blocky. Chroma is quantised with the luminance table rather than
    subsampled, so this is a stand-in for MJPEG, not a bit-exact encoder.
    """

    quality: float = 40.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.quality) and 1.0 <= self.quality <= 100.0):
            raise ValueError("quality must be between 1 and 100")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        augmented = jpeg_artifacts(video, self.quality)

        metadata.setdefault("augmentation_params", {})[self.name] = {"quality": self.quality}
        return augmented, metadata
