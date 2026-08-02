from __future__ import annotations

from typing import Any

import numpy as np


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
