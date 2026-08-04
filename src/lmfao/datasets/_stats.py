"""Feature statistics for LeRobot metadata (required for training normalization).

LeRobot normalizes observations and actions using per-feature statistics loaded
from ``meta/stats.json``; a dataset written without them fails at training time.
It stores, per feature, ``{min, max, mean, std, count, q01, q10, q50, q90,
q99}`` both aggregated (``meta/stats.json``) and per-episode (as ``stats/<feature>
/<key>`` columns in ``meta/episodes/*.parquet``).

Numeric features (state, action, timestamp, the bookkeeping indices) are reduced
exactly over their rows. Image/video features are reduced per channel over a
bounded pixel subsample per episode (normalized to ``[0, 1]``), which keeps
memory flat across a whole dataset while still giving representative
normalization stats. Image stats are shaped ``[C, 1, 1]`` to match LeRobot's
reference layout.
"""

from __future__ import annotations

import numpy as np

QUANTILES = ((0.01, "q01"), (0.1, "q10"), (0.5, "q50"), (0.9, "q90"), (0.99, "q99"))
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def reduce_samples(values: np.ndarray, *, is_image: bool, max_image_pixels: int = 4096, seed: int = 0) -> np.ndarray:
    """Reduce a feature's per-frame values to a 2D ``(M, D)`` sample matrix."""
    a = np.asarray(values)
    if is_image:
        # (F, H, W, C) uint8 -> (M, C) float in [0, 1]. Subsample while still
        # uint8: converting the whole episode first materializes an 8x float64
        # copy (tens of GB for a real episode).
        channels = a.shape[-1]
        flat = a.reshape(-1, channels)
        if flat.shape[0] > max_image_pixels:
            idx = np.random.default_rng(seed).integers(0, flat.shape[0], max_image_pixels)
            idx.sort()
            flat = flat[idx]
        return flat.astype(np.float64) / 255.0
    a = a.astype(np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return a


def stats_from_samples(reduced: np.ndarray, *, count: int, image_channels: int | None = None) -> dict[str, np.ndarray]:
    """Compute the LeRobot stat dict (arrays) from a ``(M, D)`` sample matrix."""
    out: dict[str, np.ndarray] = {
        "min": reduced.min(axis=0),
        "max": reduced.max(axis=0),
        "mean": reduced.mean(axis=0),
        "std": reduced.std(axis=0),
    }
    for p, name in QUANTILES:
        out[name] = np.quantile(reduced, p, axis=0)
    if image_channels is not None:
        out = {k: v.reshape(image_channels, 1, 1) for k, v in out.items()}
    # LeRobot stores count as a single-element list for every feature,
    # including images (never broadcast to the feature shape).
    out["count"] = np.asarray([count], dtype=np.int64)
    return out


def stats_to_lists(stats: dict[str, np.ndarray]) -> dict[str, list]:
    """Convert stat arrays to nested Python lists for JSON / parquet."""
    return {k: np.asarray(v).tolist() for k, v in stats.items()}
