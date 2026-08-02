from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import build_many


class AugmentationPipeline:
    """Composable video augmentation pipeline."""

    def __init__(self, augmenters: Sequence[Augmenter], seed: Optional[int] = None) -> None:
        self.augmenters = list(augmenters)
        self.seed = seed

    @classmethod
    def from_config(cls, configs: Iterable[dict[str, Any]], seed: Optional[int] = None) -> "AugmentationPipeline":
        return cls(build_many(configs), seed=seed)

    def __call__(self, video: Video, metadata: Optional[Mapping[str, Any]] = None) -> tuple[Video, Metadata]:
        rng = np.random.default_rng(self.seed)
        output = video
        run_metadata: Metadata = dict(metadata or {})

        applied: list[str] = []
        for augmenter in self.augmenters:
            output, run_metadata = augmenter(output, run_metadata, rng)
            applied.append(augmenter.name)

        run_metadata["augmentations"] = applied
        return output, run_metadata

