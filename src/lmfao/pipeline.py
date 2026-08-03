from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import build_augmenter


@dataclass(frozen=True)
class AugmentationStep:
    augmenter: Augmenter
    probability: float = 1.0


class AugmentationPipeline:
    """Composable video augmentation pipeline."""

    def __init__(self, steps: Sequence[Augmenter | AugmentationStep], seed: int | None = None) -> None:
        self.steps = [step if isinstance(step, AugmentationStep) else AugmentationStep(step) for step in steps]
        self.seed = seed

    @classmethod
    def from_config(cls, configs: Iterable[dict[str, Any]], seed: int | None = None) -> AugmentationPipeline:
        steps: list[AugmentationStep] = []
        for config in configs:
            item = dict(config)
            name = item.pop("name")
            params = item.pop("params", {})
            probability = float(item.pop("probability", 1.0))
            enabled = bool(item.pop("enabled", True))

            if item:
                extra = ", ".join(sorted(item))
                raise ValueError(f"unknown config fields for '{name}': {extra}")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for '{name}' must be between 0 and 1")
            if not enabled:
                continue

            steps.append(AugmentationStep(build_augmenter(name, **params), probability=probability))
        return cls(steps, seed=seed)

    def __call__(self, video: Video, metadata: Mapping[str, Any] | None = None) -> tuple[Video, Metadata]:
        rng = np.random.default_rng(self.seed)
        output = video
        # Deep-copy so seasoning never mutates the caller's (episode's) metadata.
        run_metadata: Metadata = copy.deepcopy(dict(metadata or {}))

        applied: list[str] = []
        skipped: list[str] = []
        # augmentation_params is keyed by name (last write wins), so a repeated
        # augmenter would lose its earlier application; history keeps every one
        # in order so the full transform stays reproducible from metadata.
        history: list[dict[str, Any]] = []
        for step in self.steps:
            augmenter = step.augmenter
            if rng.random() > step.probability:
                skipped.append(augmenter.name)
                continue

            output, run_metadata = augmenter(output, run_metadata, rng)
            applied.append(augmenter.name)
            params = run_metadata.get("augmentation_params", {}).get(augmenter.name)
            history.append({"name": augmenter.name, "params": copy.deepcopy(params)})

        run_metadata["augmentations"] = applied
        run_metadata["skipped_augmentations"] = skipped
        run_metadata["augmentation_history"] = history
        return output, run_metadata
