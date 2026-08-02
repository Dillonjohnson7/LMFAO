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


def to_source_frame(rgb_uint8: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, 3)`` uint8 RGB render into the reference frames'
    channel count, dtype, and value range.

    Synthetic frames must be represented exactly like the real episode's frames:
    downstream augmenters treat a float video as already scaled (``[0, 1]`` or
    ``[0, 255]``), so writing raw ``0..255`` values into a float ``[0, 1]``
    episode would make synthetic frames ~255x too bright. This is the inverse of
    :func:`to_unit_rgb`; the float-range convention is inferred from the whole
    reference array so a single dark frame can't mislead it.
    """
    reference = np.asarray(reference)
    matched = match_channels(rgb_uint8, reference.shape[-1])  # uint8, right channels
    dtype = reference.dtype

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.max == 255:
            return matched.astype(dtype)
        scaled = np.rint(matched.astype(np.float64) / 255.0 * info.max)
        return np.clip(scaled, info.min, info.max).astype(dtype)

    # Float target: match the reference's [0, 1] vs [0, 255] convention.
    hi = float(reference.max()) if reference.size else 1.0
    scale = 1.0 if hi <= 1.0 else 255.0
    return (matched.astype(np.float64) / 255.0 * scale).astype(dtype)
