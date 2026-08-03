from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from lmfao.base import Video, preserve_dtype


def apply_pointwise(video: Video, transform: Callable[[np.ndarray], np.ndarray]) -> Video:
    """Apply a per-pixel intensity ``transform`` (a float ``levels -> values`` map).

    For integer video (the uint8 case that matters) the transform depends only on
    the input byte value, so it is evaluated once on a small lookup table (256
    entries for uint8) and applied as a single gather -- no whole-clip float32
    buffer, and one pass over memory instead of ~4. Bit-identical to computing
    ``clip(transform(v.astype(float32)))`` per pixel. Float video falls back to
    the direct float path.
    """
    if np.issubdtype(video.dtype, np.integer) and video.itemsize <= 2:
        info = np.iinfo(video.dtype)
        levels = np.arange(info.min, info.max + 1, dtype=np.float32)
        lut = np.clip(transform(levels), info.min, info.max).astype(video.dtype)
        if info.min == 0:
            return lut[video]
        return lut[video.astype(np.int64) - info.min]
    return preserve_dtype(video, np.asarray(transform(video.astype(np.float32, copy=True))))


def validate_positive_range(
    minimum: float,
    maximum: float,
    minimum_name: str,
    maximum_name: str,
) -> tuple[float, float]:
    minimum = float(minimum)
    maximum = float(maximum)
    if not 0.0 < minimum <= maximum:
        raise ValueError(f"{minimum_name}/{maximum_name} must satisfy 0.0 < {minimum_name} <= {maximum_name}")
    return minimum, maximum


def sample_range(
    value: float | None,
    minimum: float,
    maximum: float,
    rng: np.random.Generator,
) -> float:
    if value is not None:
        return float(value)
    return float(rng.uniform(minimum, maximum))


def metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)
