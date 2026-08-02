from __future__ import annotations

from typing import Any

import numpy as np

_PAD_MODES = {
    "reflect": "reflect",
    "zero": "constant",
}


def pad_mode_for_numpy(pad_mode: str) -> str:
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"pad_mode must be one of {sorted(_PAD_MODES)}, got {pad_mode!r}")
    return _PAD_MODES[pad_mode]


def sample_shift(
    value: int | None,
    pad: int,
    num_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if value is None:
        return rng.integers(-pad, pad + 1, size=num_frames)
    return np.full(num_frames, int(np.clip(value, -pad, pad)))


def metadata_params(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)
