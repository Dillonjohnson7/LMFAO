from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lmfao.base import Video

FillMode = str | tuple[int, int, int]


def validate_range(values: Sequence[float], field_name: str, *, allow_zero: bool = False) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    lower = float(values[0])
    upper = float(values[1])
    minimum = 0.0 if allow_zero else np.finfo(float).eps
    if lower < minimum or upper < minimum:
        expectation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} values must be {expectation}")
    if lower > upper:
        raise ValueError(f"{field_name} minimum cannot exceed maximum")
    return lower, upper


def validate_fraction(value: float, field_name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{field_name} must be greater than 0 and at most 1")
    return value


def validate_fraction_range(values: Sequence[float], field_name: str) -> tuple[float, float]:
    lower, upper = validate_range(values, field_name)
    if upper > 1.0:
        raise ValueError(f"{field_name} values must be at most 1")
    return lower, upper


def validate_fill(fill: FillMode) -> FillMode:
    if isinstance(fill, str):
        if fill not in {"black", "white", "mean", "random_color"}:
            raise ValueError("fill must be one of: black, white, mean, random_color")
        return fill
    if len(fill) != 3:
        raise ValueError("RGB fill tuples must have exactly three channels")
    channels = tuple(int(channel) for channel in fill)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("RGB fill channels must be between 0 and 255")
    return channels


def sample_box(
    height: int,
    width: int,
    area_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, int]:
    target_area = float(rng.uniform(*area_range)) * height * width
    aspect_ratio = float(rng.uniform(*aspect_ratio_range))
    box_height = max(1, min(height, int(round(np.sqrt(target_area / aspect_ratio)))))
    box_width = max(1, min(width, int(round(np.sqrt(target_area * aspect_ratio)))))
    top = int(rng.integers(0, height - box_height + 1))
    left = int(rng.integers(0, width - box_width + 1))
    return {"top": top, "left": left, "height": box_height, "width": box_width}


def fill_region(region: Video, fill: FillMode, rng: np.random.Generator) -> None:
    channels = region.shape[-1]
    if fill == "black":
        region[...] = 0
        return
    if fill == "white":
        region[...] = _white_value(region.dtype)
        return
    if fill == "mean":
        region[...] = region.mean(axis=tuple(range(region.ndim - 1)), keepdims=True)
        return
    if fill == "random_color":
        region[...] = _random_color(region.dtype, channels, rng)
        return
    if channels != 3:
        raise ValueError("RGB fill tuples require RGB video input")
    region[...] = np.asarray(fill, dtype=region.dtype)


def metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)


def _white_value(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    return 1.0


def _random_color(dtype: np.dtype, channels: int, rng: np.random.Generator) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        high = int(np.iinfo(dtype).max) + 1
        return rng.integers(0, high, size=(channels,), dtype=dtype)
    return rng.random(size=(channels,)).astype(dtype, copy=False)
