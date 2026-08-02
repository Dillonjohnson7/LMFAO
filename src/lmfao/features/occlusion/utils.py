from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FillMode = str | tuple[int, int, int]


def validate_probability(value: float, field_name: str = "probability") -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def validate_range(values: Sequence[float], field_name: str, *, positive: bool = True) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    lower = float(values[0])
    upper = float(values[1])
    if positive and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(f"{field_name} values must be positive")
    if lower > upper:
        raise ValueError(f"{field_name} minimum cannot exceed maximum")
    return lower, upper


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


def copy_metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)
