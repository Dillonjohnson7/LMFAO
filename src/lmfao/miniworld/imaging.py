from __future__ import annotations

import numpy as np


def to_unit_rgb(frame: np.ndarray) -> np.ndarray:
    """Normalise a frame to ``(H, W, 3)`` float in ``[0, 1]``.

    Handles integer or float input and grayscale / RGB / RGBA channel counts.
    Alpha is dropped; grayscale is broadcast to three channels.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError("frame must be (H, W, C)")

    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
        unit = arr.astype(float) / max(scale, 1.0)
    else:
        unit = arr.astype(float)
        if unit.size and unit.max() > 1.0:
            unit = unit / 255.0

    channels = unit.shape[-1]
    if channels == 1:
        unit = np.repeat(unit, 3, axis=-1)
    elif channels == 4:
        unit = unit[:, :, :3]
    elif channels != 3:
        raise ValueError(f"unsupported channel count: {channels}")
    return np.clip(unit, 0.0, 1.0)


def match_channels(rgb_uint8: np.ndarray, channels: int) -> np.ndarray:
    """Convert an ``(H, W, 3)`` uint8 RGB frame to the requested channel count."""
    if channels == 3:
        return rgb_uint8
    if channels == 1:
        gray = (0.299 * rgb_uint8[:, :, 0] + 0.587 * rgb_uint8[:, :, 1] + 0.114 * rgb_uint8[:, :, 2])
        return np.clip(gray + 0.5, 0, 255).astype(np.uint8)[:, :, None]
    if channels == 4:
        alpha = np.full(rgb_uint8.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([rgb_uint8, alpha], axis=-1)
    raise ValueError(f"unsupported channel count: {channels}")
