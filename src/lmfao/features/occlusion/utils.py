from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lmfao.base import Video

FillMode = str


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
    if fill != "black":
        raise ValueError("fill must be 'black'")
    return fill


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
    # On strongly non-square frames one dimension gets capped by the frame,
    # which would silently shrink the realized area far below the sampled
    # target. When that happens, grow the other dimension to recover the area.
    if box_height * box_width < target_area:
        box_width = max(1, min(width, int(round(target_area / box_height))))
        box_height = max(1, min(height, int(round(target_area / box_width))))
    top = int(rng.integers(0, height - box_height + 1))
    left = int(rng.integers(0, width - box_width + 1))
    return {"top": top, "left": left, "height": box_height, "width": box_width}


def fill_region(region: Video, fill: FillMode) -> None:
    validate_fill(fill)
    if region.shape[-1] == 4:
        # Black out the color channels but keep the occluder opaque, otherwise an
        # RGBA occluder would be fully transparent (alpha 0) instead of black.
        region[..., :3] = 0
        region[..., 3] = np.iinfo(region.dtype).max if np.issubdtype(region.dtype, np.integer) else 1
    else:
        region[...] = 0


def metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)

