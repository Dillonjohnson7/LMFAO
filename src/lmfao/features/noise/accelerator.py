"""Accelerated sampling for the noise features.

Every device-specific detail for noise lives in this module. The feature modules
next to it only name a distribution and a strength; they never import torch,
branch on a device, or see anything other than a NumPy array. That keeps the
rest of LMFAO free of optional GPU dependencies.

Drawing the random numbers, not the arithmetic that follows, is what makes noise
expensive. On a 32-frame 720p batch this module is roughly 30x faster than NumPy
on Apple Metal, and it stays about that much faster even though it uploads the
clip and downloads the result on every call. Keeping the video resident on the
device between calls would only buy another ~20%, which is not worth leaking
device handles through the public contract.

Falls back to NumPy whenever no accelerator is present or the video's dtype is
one the device cannot represent, so callers never have to check first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lmfao.base import Video, preserve_dtype

# Dtypes both torch and MPS handle. Notably absent is float64, which Metal
# cannot represent, and the wider unsigned integer types, which torch lacks.
_ACCELERATED_DTYPES = frozenset({"uint8", "int8", "int16", "int32", "float16", "float32"})

_UNAVAILABLE = {
    "cuda": "device='cuda' requested but no CUDA device is available to PyTorch.",
    "mps": "device='mps' requested but Apple Metal is not available to PyTorch.",
}


def add_noise(
    video: Video,
    distribution: str,
    strength: float,
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[Video, str]:
    """Add scaled ``distribution`` noise to ``video``, preserving shape and dtype.

    ``strength`` is a fraction of the dtype's dynamic range, so the same value
    means the same visible amount of noise for uint8 and float video.

    Returns the augmented video and the device that actually ran, which callers
    record in metadata so an augmented clip stays auditable.
    """

    target = resolve_device(device, video)
    if target == "cpu":
        return _add_noise_numpy(video, distribution, strength, rng), target

    return _add_noise_torch(video, distribution, strength, rng, target), target


def resolve_device(device: str = "auto", video: Video | None = None) -> str:
    """Pick the device to sample on.

    ``"auto"`` prefers an accelerator and quietly falls back to the CPU. Naming a
    device explicitly raises if it is unavailable, so a typo or a missing driver
    fails loudly instead of silently costing 30x.
    """

    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"unknown device '{device}'. Expected 'auto', 'cpu', 'cuda', or 'mps'")

    if device == "cpu":
        return "cpu"

    if video is not None and video.dtype.name not in _ACCELERATED_DTYPES:
        if device == "auto":
            return "cpu"
        raise ValueError(f"device='{device}' cannot sample for dtype {video.dtype}; use device='cpu'")

    torch = _torch()
    if torch is None:
        if device == "auto":
            return "cpu"
        raise ImportError(f"device='{device}' needs PyTorch. Install it with `pip install torch`.")

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if device == "auto":
        return next((name for name, ready in available.items() if ready), "cpu")
    if not available[device]:
        raise RuntimeError(_UNAVAILABLE[device])

    return device


def dynamic_range(dtype: np.dtype) -> float:
    """The span a strength of ``1.0`` covers for ``dtype``."""

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.max) - float(info.min)

    return 1.0


def _intensity_scale(video: Video) -> np.ndarray:
    """The square root of normalised intensity, which is how photon noise grows.

    Scaling a unit-variance draw by this turns signal-independent grain into
    shot noise: bright pixels get noisy, dark ones stay comparatively clean.
    """

    scale = np.asarray(video, dtype=np.float32) / dynamic_range(video.dtype)
    np.clip(scale, 0.0, 1.0, out=scale)
    return np.sqrt(scale, out=scale)


def _add_noise_numpy(video: Video, distribution: str, strength: float, rng: np.random.Generator) -> Video:
    """Sample the whole clip in one call, reusing the buffer for the arithmetic."""

    if distribution == "uniform":
        noise = rng.random(video.shape, dtype=np.float32)
        noise -= 0.5
    else:
        noise = rng.standard_normal(video.shape, dtype=np.float32)

    if distribution == "shot":
        noise *= _intensity_scale(video)

    noise *= strength * dynamic_range(video.dtype)
    noise += video

    # preserve_dtype casts, and casting truncates toward zero. Rounding first
    # keeps the noise zero-mean rather than biasing every frame darker.
    if np.issubdtype(video.dtype, np.integer):
        np.rint(noise, out=noise)

    return preserve_dtype(video, noise)


def _add_noise_torch(video: Video, distribution: str, strength: float, rng: np.random.Generator, device: str) -> Video:
    """The same arithmetic on an accelerator, returning a NumPy array.

    The device generator is seeded from ``rng`` so randomness still flows from
    the pipeline's seed and a seeded run stays reproducible. The device and the
    CPU draw different values for the same seed, because they use different
    generators; each is reproducible against itself.
    """

    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(_device_seed(rng))
    frames = torch.as_tensor(video, device=device)
    shape = tuple(video.shape)

    if distribution == "uniform":
        noise = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
        noise -= 0.5
    else:
        noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    if distribution == "shot":
        noise *= (frames / dynamic_range(video.dtype)).clamp_(0.0, 1.0).sqrt_()

    noise *= strength * dynamic_range(video.dtype)
    noise += frames

    if np.issubdtype(video.dtype, np.integer):
        info = np.iinfo(video.dtype)
        noise.round_().clamp_(float(info.min), float(info.max))

    return noise.to(getattr(torch, video.dtype.name)).cpu().numpy()


def _device_seed(rng: np.random.Generator) -> int:
    """Draw a device generator seed from the pipeline's rng."""

    return int(rng.integers(np.iinfo(np.int64).max))


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None

    return torch
