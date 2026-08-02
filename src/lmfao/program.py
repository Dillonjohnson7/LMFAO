"""The parallel-switch driver: real + (optional) synthetic -> seasoned training set.

This wires the two halves of LMFAO together exactly as the plan's §1 diagram and
§8f config schema describe. One combined config drives both:

    {
        "miniworld": {"enabled": True, "n_synthetic": 40, ...},   # GENERATE
        "pipeline":  [ {"name": "lighting.color_temperature", ...} ],  # ADJUST
    }

The miniworld branch is a *source* running in parallel with the real-data path,
gated by ``miniworld.enabled``. When off, the result is byte-for-byte the v1
pipeline over the real episodes. When on, synthetic episodes are manufactured,
merged with the real ones, and every episode -- real or synthetic -- is seasoned
by the same pixel pipeline. The pipeline never learns which is which; provenance
lives only in each episode's metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lmfao.datasets import Episode
from lmfao.miniworld.config import MiniWorldConfig
from lmfao.miniworld.generator import MiniWorldGenerator
from lmfao.pipeline import AugmentationPipeline


@dataclass
class TrainingSet:
    """The merged, seasoned output of :func:`generate_training_set`."""

    episodes: list[Episode] = field(default_factory=list)

    @property
    def num_real(self) -> int:
        return sum(1 for ep in self.episodes if not ep.is_synthetic)

    @property
    def num_synthetic(self) -> int:
        return sum(1 for ep in self.episodes if ep.is_synthetic)

    @property
    def num_frames(self) -> int:
        return sum(ep.num_frames for ep in self.episodes)

    def frames(self) -> np.ndarray:
        from lmfao.datasets import stack_frames

        return stack_frames(self.episodes)

    def summary(self) -> dict[str, int]:
        return {
            "episodes": len(self.episodes),
            "real": self.num_real,
            "synthetic": self.num_synthetic,
            "frames": self.num_frames,
        }


def generate_training_set(
    real_episodes: Sequence[Episode],
    config: Mapping[str, Any],
    seed: int | None = None,
    generator: MiniWorldGenerator | None = None,
) -> TrainingSet:
    """Run the full switch: generate, merge, and season.

    Parameters
    ----------
    real_episodes:
        The recorded episodes (the always-on lane).
    config:
        Combined config with optional ``"miniworld"`` and ``"pipeline"`` keys.
    seed:
        Base seed for reproducible per-episode pipeline seasoning. Each episode
        gets ``seed + index`` so real and synthetic frames are seasoned
        differently while the whole run stays reproducible.
    generator:
        Inject a pre-built :class:`MiniWorldGenerator` (e.g. with a real splat
        backend). If omitted, one is built from the config's miniworld block
        using the numpy reference backend.
    """
    real = list(real_episodes)
    # A missing miniworld block means the switch is simply not present, so the
    # synthetic branch stays dark and the result is pure v1. Only an explicit
    # block can turn generation on (and it still honours its own `enabled`).
    mw_block = config.get("miniworld")
    mw_config = MiniWorldConfig.from_dict(mw_block) if mw_block else MiniWorldConfig(enabled=False)
    pipeline_config = list(config.get("pipeline", []) or [])

    synthetic: list[Episode] = []
    if mw_config.enabled:
        gen = generator or MiniWorldGenerator(mw_config)
        synthetic = gen.generate(real)

    merged = real + synthetic
    seasoned = [
        _season(episode, pipeline_config, _derive_seed(seed, index))
        for index, episode in enumerate(merged)
    ]
    return TrainingSet(episodes=seasoned)


def _derive_seed(seed: int | None, index: int) -> int | None:
    return None if seed is None else seed + index


def _season(episode: Episode, pipeline_config: list, seed: int | None) -> Episode:
    """Run the pixel pipeline over one episode's frames, preserving provenance."""
    if not pipeline_config:
        return episode
    pipeline = AugmentationPipeline.from_config(pipeline_config, seed=seed)
    frames, metadata = pipeline(episode.frames, episode.metadata)
    return episode.with_frames(frames, metadata=dict(metadata))
