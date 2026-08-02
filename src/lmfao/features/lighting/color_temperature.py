from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video, preserve_dtype
from lmfao.features.lighting.utils import metadata_params, sample_range
from lmfao.registry import register_augmenter


@register_augmenter(
    "lighting.color_temperature",
    tags=("lighting", "color", "color-temperature"),
    description=(
        "Shifts RGB channel balance along a warm<->cool axis to mimic different "
        "ambient light sources: warm/red-shifted overhead lighting at night vs. "
        "cool/blue-shifted daylight through a window at midday."
    ),
)
@dataclass
class ColorTemperatureShift(Augmenter):
    """Simulates ambient light color casts via per-channel gain."""

    shift: float | None = None
    min_shift: float = -1.0
    max_shift: float = 1.0
    intensity: float = 0.35

    def __post_init__(self) -> None:
        self.min_shift = float(self.min_shift)
        self.max_shift = float(self.max_shift)
        if not -1.0 <= self.min_shift <= self.max_shift <= 1.0:
            raise ValueError("min_shift/max_shift must satisfy -1.0 <= min_shift <= max_shift <= 1.0")
        if self.shift is not None:
            self.shift = float(np.clip(self.shift, -1.0, 1.0))
        if self.intensity < 0.0:
            raise ValueError("intensity must be non-negative")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        shift = float(np.clip(sample_range(self.shift, self.min_shift, self.max_shift, rng), -1.0, 1.0))

        augmented = video.astype(np.float32, copy=True)
        channels = augmented.shape[-1]
        applied_to_color = channels >= 3

        red_gain = 1.0
        blue_gain = 1.0
        if applied_to_color:
            red_gain = 1.0 + self.intensity * -shift
            blue_gain = 1.0 + self.intensity * shift
            augmented[..., 0] *= red_gain
            augmented[..., 2] *= blue_gain

        augmented = preserve_dtype(video, augmented)

        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(
            {
                "shift": shift,
                "red_gain": red_gain,
                "blue_gain": blue_gain,
                "intensity": self.intensity,
                "applied_to_color_channels": applied_to_color,
            }
        )
        return augmented, metadata
