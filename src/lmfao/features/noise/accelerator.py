"""Accelerated sampling for the noise features.

Every device-specific detail for noise lives in this module. The feature modules
next to it only name a distribution and a strength; they never import torch,
branch on a device, or see anything other than a NumPy array. That keeps the
rest of LMFAO free of optional GPU dependencies.

Drawing the random numbers, not the arithmetic that follows, is what makes noise
expensive, and NumPy draws them on one core. So this module parallelises the CPU
path over a fixed set of blocks as well as offering a device: on a 32-frame 720p
clip that took the CPU from ~470 ms to ~70 ms, and Apple Metal runs the same
clip in ~19 ms.

That leaves the accelerator roughly 4x ahead of the CPU rather than the ~25x it
led by when the CPU path was single-threaded, which is worth knowing before
reaching for a GPU: the device is no longer the only way to make noise cheap,
and it stays unavailable wherever torch cannot be installed. Metal keeps that
lead even though it uploads the clip and downloads the result every call --
keeping the video resident between calls would buy only another ~20%, which is
not worth leaking device handles through the public contract.

Falls back to NumPy whenever no accelerator is present or the video's dtype is
one the device cannot represent, so callers never have to check first.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from lmfao.base import Video

# Dtypes both torch and MPS handle. Notably absent is float64, which Metal
# cannot represent, and the wider unsigned integer types, which torch lacks.
_ACCELERATED_DTYPES = frozenset({"uint8", "int8", "int16", "int32", "float16", "float32"})

_UNAVAILABLE = {
    "cuda": "device='cuda' requested but no CUDA device is available to PyTorch.",
    "mps": "device='mps' requested but Apple Metal is not available to PyTorch.",
}

# How many pieces CPU work is cut into. Fixed rather than tied to the host's
# core count so that a seed produces the same pixels on a laptop and on a pod;
# only *whether* the pieces run concurrently depends on the machine.
_BLOCKS = 16

# Below roughly a megapixel a thread pool costs more than the work it saves.
_PARALLEL_MIN_ELEMENTS = 1 << 20


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


def run_blocks(fill: Callable[[int], None], blocks: int, elements: int) -> None:
    """Call ``fill`` for every block index, concurrently once it is worth it.

    NumPy releases the GIL for the bulk array work inside a block, so threads
    buy real parallelism here without the copying a process pool would cost.
    Running the same blocks serially produces byte-identical output, so the
    threshold below changes only the speed, never the result.
    """

    if blocks <= 1 or elements < _PARALLEL_MIN_ELEMENTS:
        for index in range(blocks):
            fill(index)
        return

    with ThreadPoolExecutor(min(blocks, os.cpu_count() or 1)) as pool:
        for _ in pool.map(fill, range(blocks)):  # consume, so a block's error surfaces
            pass


def block_bounds(length: int, blocks: int = _BLOCKS) -> np.ndarray:
    """Split ``length`` into ``blocks`` contiguous spans."""

    return np.linspace(0, length, blocks + 1).astype(np.intp)


def store_as(destination: np.ndarray, values: np.ndarray) -> None:
    """Round and clip float ``values`` into an integer ``destination``, then write.

    Casting truncates toward zero, so rounding first is what keeps a zero-mean
    perturbation from biasing every frame darker.
    """

    if np.issubdtype(destination.dtype, np.integer):
        info = np.iinfo(destination.dtype)
        np.rint(values, out=values)
        np.clip(values, info.min, info.max, out=values)
    destination[...] = values


def _intensity_scale(video: Video) -> np.ndarray:
    """The square root of normalised intensity, which is how photon noise grows.

    Scaling a unit-variance draw by this turns signal-independent grain into
    shot noise: bright pixels get noisy, dark ones stay comparatively clean.
    """

    scale = np.asarray(video, dtype=np.float32) / dynamic_range(video.dtype)
    np.clip(scale, 0.0, 1.0, out=scale)
    return np.sqrt(scale, out=scale)


def block_generators(rng: np.random.Generator, blocks: int = _BLOCKS) -> list[np.random.Generator]:
    """Independent per-block streams, still descended from the pipeline's rng."""

    return [np.random.default_rng(seed) for seed in np.random.SeedSequence(_device_seed(rng)).spawn(blocks)]


def _add_noise_numpy(video: Video, distribution: str, strength: float, rng: np.random.Generator) -> Video:
    """Fill the output one block at a time, each block start to finish.

    Two things make this much faster than sampling the whole clip in one call.
    Drawing the random numbers is about 70% of the work and NumPy's generator is
    single-threaded, so blocks are drawn on their own streams in parallel. And
    each block scales, adds, rounds and casts while it is still hot in cache,
    which replaces five passes over a whole-clip float32 buffer with one pass
    over a small one.

    Only the blocks in flight hold float32 scratch, so peak memory drops from
    about nine bytes per pixel to four -- which is why a clip that OOMed can fit.
    """

    source = np.ravel(video)
    out = np.empty(source.shape, dtype=video.dtype)
    bounds = block_bounds(source.size)
    generators = block_generators(rng)
    scale = strength * dynamic_range(video.dtype)

    def fill(index: int) -> None:
        low, high = bounds[index], bounds[index + 1]
        if low == high:
            return
        block = source[low:high]
        block_rng = generators[index]

        if distribution == "uniform":
            noise = block_rng.random(high - low, dtype=np.float32)
            noise -= 0.5
        else:
            noise = block_rng.standard_normal(high - low, dtype=np.float32)

        if distribution == "shot":
            noise *= _intensity_scale(block)

        noise *= scale
        noise += block
        store_as(out[low:high], noise)

    run_blocks(fill, _BLOCKS, source.size)
    return out.reshape(video.shape)


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
