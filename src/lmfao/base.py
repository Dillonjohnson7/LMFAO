from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

Video = np.ndarray
Metadata = MutableMapping[str, Any]


class Augmenter(ABC):
    """Base class for all video augmenters.

    Implementations should accept and return a NumPy array with shape
    ``(frames, height, width, channels)``. Metadata is passed through the
    pipeline so labels, annotations, or audit information can be updated by
    augmenters that need it.
    """

    name: str

    def __call__(
        self,
        video: Video,
        metadata: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[Video, Metadata]:
        self.validate_video(video)
        run_rng = rng if rng is not None else np.random.default_rng()
        run_metadata: Metadata = dict(metadata or {})
        return self.apply(video, run_metadata, run_rng)

    @abstractmethod
    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        """Apply the augmentation."""

    @staticmethod
    def validate_video(video: Video) -> None:
        if not isinstance(video, np.ndarray):
            raise TypeError("video must be a NumPy array")
        if video.ndim != 4:
            raise ValueError("video must have shape (frames, height, width, channels)")
        if video.shape[-1] not in (1, 3, 4):
            raise ValueError("video channels must be grayscale, RGB, or RGBA")


def preserve_dtype(original: Video, augmented: Video) -> Video:
    """Clip and cast augmented frames back to the original dtype."""

    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        return np.clip(augmented, info.min, info.max).astype(original.dtype)

    return augmented.astype(original.dtype, copy=False)
