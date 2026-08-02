from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video, preserve_dtype
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
    """Simulates ambient light color casts via per-channel gain.

    ``shift`` runs on a [-1, 1] warm<->cool axis:
      -1.0 -> fully warm/red-shifted (e.g. indoor overhead lighting at night)
       0.0 -> neutral, no color cast
      +1.0 -> fully cool/blue-shifted (e.g. midday sunlight through a window)

    If ``shift`` is left as ``None``, a value is sampled uniformly from
    ``[min_shift, max_shift]`` on every call using the pipeline's rng.
    """

    shift: float | None = None
    min_shift: float = -1.0
    max_shift: float = 1.0
    intensity: float = 0.35

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        if not -1.0 <= self.min_shift <= self.max_shift <= 1.0:
            raise ValueError("min_shift/max_shift must satisfy -1.0 <= min_shift <= max_shift <= 1.0")

        effective_shift = self.shift
        if effective_shift is None:
            effective_shift = float(rng.uniform(self.min_shift, self.max_shift))
        effective_shift = float(np.clip(effective_shift, -1.0, 1.0))

        augmented = video.astype(np.float32, copy=True)
        channels = augmented.shape[-1]
        applied_to_color = channels >= 3

        red_gain = 1.0
        blue_gain = 1.0
        if applied_to_color:
            red_gain = 1.0 + self.intensity * -effective_shift
            blue_gain = 1.0 + self.intensity * effective_shift
            augmented[..., 0] *= red_gain
            augmented[..., 2] *= blue_gain

        augmented = preserve_dtype(video, augmented)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "shift": effective_shift,
            "red_gain": red_gain,
            "blue_gain": blue_gain,
            "intensity": self.intensity,
            "applied_to_color_channels": applied_to_color,
        }
        return augmented, metadata


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
    """Multiplicative exposure gain.

    ``factor`` < 1.0 darkens the frame, ``factor`` > 1.0 brightens it,
    ``factor`` == 1.0 is a no-op. If left as ``None``, a value is sampled
    uniformly from ``[min_factor, max_factor]`` on every call using the
    pipeline's rng.
    """

    factor: float | None = None
    min_factor: float = 0.6
    max_factor: float = 1.4

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        if not 0.0 < self.min_factor <= self.max_factor:
            raise ValueError("min_factor/max_factor must satisfy 0.0 < min_factor <= max_factor")

        effective_factor = self.factor
        if effective_factor is None:
            effective_factor = float(rng.uniform(self.min_factor, self.max_factor))
        if effective_factor <= 0.0:
            raise ValueError("factor must be > 0.0")

        augmented = video.astype(np.float32, copy=True) * effective_factor
        augmented = preserve_dtype(video, augmented)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "factor": effective_factor,
        }
        return augmented, metadata


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
    """Multiplicative contrast adjustment pivoted around mid-gray.

    ``factor`` < 1.0 flattens contrast (washed out), ``factor`` > 1.0
    sharpens it (harsher highlights/shadows), ``factor`` == 1.0 is a no-op.
    If left as ``None``, a value is sampled uniformly from
    ``[min_factor, max_factor]`` on every call using the pipeline's rng.
    """

    factor: float | None = None
    min_factor: float = 0.6
    max_factor: float = 1.6

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        if not 0.0 < self.min_factor <= self.max_factor:
            raise ValueError("min_factor/max_factor must satisfy 0.0 < min_factor <= max_factor")

        effective_factor = self.factor
        if effective_factor is None:
            effective_factor = float(rng.uniform(self.min_factor, self.max_factor))
        if effective_factor <= 0.0:
            raise ValueError("factor must be > 0.0")

        if np.issubdtype(video.dtype, np.integer):
            info = np.iinfo(video.dtype)
            pivot = (float(info.min) + float(info.max)) / 2.0
        else:
            pivot = 0.5

        augmented = (video.astype(np.float32, copy=True) - pivot) * effective_factor + pivot
        augmented = preserve_dtype(video, augmented)

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "factor": effective_factor,
            "pivot": pivot,
        }
        return augmented, metadata
