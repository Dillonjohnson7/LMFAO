from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video, preserve_dtype
from lmfao.features.lighting.utils import metadata_params, sample_range
from lmfao.registry import register_augmenter


def _apply_channel_gains(video: Video, red_gain: float, blue_gain: float) -> Video:
    """Scale the red (0) and blue (2) channels, leaving the rest untouched.

    Integer video uses a per-channel lookup table (a single uint8 pass, no
    whole-clip float buffer); float video falls back to the direct path. Both
    are bit-identical to the old ``astype(float32); ch *= gain; preserve_dtype``.
    """
    if np.issubdtype(video.dtype, np.integer) and video.itemsize <= 2:
        info = np.iinfo(video.dtype)
        levels = np.arange(info.min, info.max + 1, dtype=np.float32)

        def lut_for(gain: float) -> np.ndarray:
            return np.clip(levels * gain, info.min, info.max).astype(video.dtype)

        def gather(channel: np.ndarray, lut: np.ndarray) -> np.ndarray:
            return lut[channel] if info.min == 0 else lut[channel.astype(np.int64) - info.min]

        out = np.empty_like(video)
        red_lut, blue_lut = lut_for(red_gain), lut_for(blue_gain)
        for c in range(video.shape[-1]):
            if c == 0:
                out[..., 0] = gather(video[..., 0], red_lut)
            elif c == 2:
                out[..., 2] = gather(video[..., 2], blue_lut)
            else:
                out[..., c] = video[..., c]
        return out

    augmented = video.astype(np.float32, copy=True)
    augmented[..., 0] *= red_gain
    augmented[..., 2] *= blue_gain
    return preserve_dtype(video, augmented)


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
            if not math.isfinite(self.shift):
                raise ValueError("shift must be a finite number")
            self.shift = float(np.clip(self.shift, -1.0, 1.0))
        if not (math.isfinite(self.intensity) and self.intensity >= 0.0):
            raise ValueError("intensity must be a non-negative, finite number")

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        shift = float(np.clip(sample_range(self.shift, self.min_shift, self.max_shift, rng), -1.0, 1.0))

        channels = video.shape[-1]
        applied_to_color = channels >= 3

        red_gain = 1.0
        blue_gain = 1.0
        if applied_to_color:
            red_gain = 1.0 + self.intensity * -shift
            blue_gain = 1.0 + self.intensity * shift

        augmented = _apply_channel_gains(video, red_gain, blue_gain) if applied_to_color else video.copy()

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
