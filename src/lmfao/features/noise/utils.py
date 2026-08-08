"""Whole-neighbourhood machinery for the noise features that are not per-pixel draws.

Gaussian, uniform and shot noise are independent draws per pixel, so they live in
`accelerator.py` and differ only by the distribution they name. Blur and
compression instead read a pixel's neighbours -- a convolution and an 8x8 block
transform -- which no random generator can express, so they share this module.

Everything here is NumPy only, matching the core install.
"""

from __future__ import annotations

import numpy as np

from lmfao.base import Video, preserve_dtype
from lmfao.features.noise.accelerator import dynamic_range

# The luminance quantisation table from the JPEG standard (Annex K). Applied to
# every channel: chroma is not subsampled, so this stands in for MJPEG rather
# than reproducing an encoder bit-for-bit.
_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def _dct_matrix() -> np.ndarray:
    """The orthonormal 8-point DCT-II as a matrix, so a block transform is `C @ B @ C.T`."""

    k = np.arange(8, dtype=np.float32)
    matrix = 0.5 * np.cos(np.pi * (2.0 * k[None, :] + 1.0) * k[:, None] / 16.0)
    matrix[0] /= np.sqrt(2.0)
    return matrix.astype(np.float32)


_DCT = _dct_matrix()


def _finish(video: Video, frames: np.ndarray) -> Video:
    """Round before the cast, which truncates, so a flat region survives intact."""

    if np.issubdtype(video.dtype, np.integer):
        np.rint(frames, out=frames)
    return preserve_dtype(video, frames)


def gaussian_kernel(radius: float) -> np.ndarray:
    """A normalised 1-D Gaussian, truncated at three sigma where the tail stops mattering."""

    half = max(1, int(np.ceil(3.0 * radius)))
    offsets = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / radius) ** 2)
    return kernel / kernel.sum()


def blur_axis(frames: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve along one ``axis``, holding the edge pixels rather than darkening them."""

    half = len(kernel) // 2
    moved = np.moveaxis(frames, axis, 0)
    padded = np.pad(moved, [(half, half)] + [(0, 0)] * (moved.ndim - 1), mode="edge")
    length = moved.shape[0]
    blurred = sum(weight * padded[i : i + length] for i, weight in enumerate(kernel))
    return np.moveaxis(blurred, 0, axis)


def defocus_blur(video: Video, radius: float) -> Video:
    """Blur every frame with an isotropic Gaussian point spread.

    The kernel is separable, so this is two 1-D passes costing O(k) per pixel
    instead of the O(k^2) a square kernel would.
    """

    kernel = gaussian_kernel(radius)
    frames = np.asarray(video, dtype=np.float32)
    frames = blur_axis(blur_axis(frames, kernel, 1), kernel, 2)
    return _finish(video, frames)


def quantization_table(quality: float) -> np.ndarray:
    """Scale the JPEG table the way libjpeg does: lower quality, coarser steps."""

    scale = 5000.0 / quality if quality < 50.0 else 200.0 - 2.0 * quality
    return np.clip(np.floor((_QUANTIZATION * scale + 50.0) / 100.0), 1.0, 255.0)


def jpeg_artifacts(video: Video, quality: float) -> Video:
    """Round-trip every frame through JPEG's 8x8 block quantisation.

    Working in the codec's own 0-255 space keeps the standard table meaningful
    for float video too, and edge-padding to a block multiple keeps frame sizes
    that are not divisible by 8 from shifting the block grid.
    """

    span = dynamic_range(video.dtype)
    frames = np.asarray(video, dtype=np.float32) * (255.0 / span)
    _, height, width, _ = frames.shape
    pad_h, pad_w = -height % 8, -width % 8
    if pad_h or pad_w:
        frames = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    frames -= 128.0  # the level shift the standard applies before transforming
    blocks = _to_blocks(frames)
    table = quantization_table(quality)
    coefficients = _DCT @ blocks @ _DCT.T
    np.rint(coefficients / table, out=coefficients)
    coefficients *= table
    frames = _from_blocks(_DCT.T @ coefficients @ _DCT, frames.shape)
    frames += 128.0

    return _finish(video, frames[:, :height, :width, :] * (span / 255.0))


def _to_blocks(frames: np.ndarray) -> np.ndarray:
    """Regroup ``(F, H, W, C)`` into ``(F, H/8, W/8, C, 8, 8)`` so matmul hits each block."""

    f, h, w, c = frames.shape
    return frames.reshape(f, h // 8, 8, w // 8, 8, c).transpose(0, 1, 3, 5, 2, 4)


def _from_blocks(blocks: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The inverse of :func:`_to_blocks`."""

    return blocks.transpose(0, 1, 4, 2, 5, 3).reshape(shape)
