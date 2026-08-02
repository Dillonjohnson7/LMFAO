from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from lmfao.base import KernelFeature, KernelRuntime, Metadata, Video
from lmfao.registry import build_kernel_feature


@dataclass(frozen=True)
class KernelStep:
    feature: KernelFeature
    probability: float = 1.0


class KernelPipeline:
    """Lightweight GPU kernel launch pipeline."""

    def __init__(self, steps: Sequence[Union[KernelFeature, KernelStep]], seed: Optional[int] = None) -> None:
        self.steps = [step if isinstance(step, KernelStep) else KernelStep(step) for step in steps]
        self.seed = seed

    @classmethod
    def from_config(cls, configs: Iterable[dict[str, Any]], seed: Optional[int] = None) -> "KernelPipeline":
        steps: list[KernelStep] = []
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

            steps.append(KernelStep(build_kernel_feature(name, **params), probability=probability))
        return cls(steps, seed=seed)

    def __call__(
        self,
        video: Video,
        runtime: KernelRuntime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Video, Metadata]:
        rng = random.Random(self.seed)
        run_metadata: Metadata = dict(metadata or {})

        applied: list[str] = []
        skipped: list[str] = []
        for step in self.steps:
            feature = step.feature
            if rng.random() > step.probability:
                skipped.append(feature.name)
                continue

            run_metadata = feature(video, runtime, run_metadata)
            applied.append(feature.name)

        run_metadata["augmentations"] = applied
        run_metadata["skipped_augmentations"] = skipped
        return video, run_metadata
