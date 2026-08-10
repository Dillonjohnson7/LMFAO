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

from collections.abc import Iterator, Mapping, Sequence
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
    # Reject unknown top-level keys outright: a typo like "minworld" would
    # otherwise silently disable generation while the run reports success.
    unknown = set(config) - {"miniworld", "pipeline"}
    if unknown:
        extra = ", ".join(sorted(unknown))
        raise ValueError(
            f"unknown top-level config keys: {extra} (expected 'miniworld' and/or 'pipeline')"
        )
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


def augment_episodes(
    episodes: Sequence[Episode],
    pipeline_config: Sequence[Mapping[str, Any]],
    *,
    variants: int = 1,
    seed: int | None = None,
    include_original: bool = False,
    original_copies: int = 1,
    index_base: int = 0,
) -> TrainingSet:
    """Apply the ADJUST pixel pipeline to real episodes (no miniworld synthesis).

    Unlike :func:`generate_training_set`, this never reconstructs or re-renders:
    it seasons the real frames at their native resolution, so the output is real
    footage, just relit / occluded / re-cropped. ``variants`` produces that many
    independently-seasoned copies of each episode (each with its own RNG stream),
    which is how one recorded demo becomes several training clips.

    Parameters
    ----------
    variants:
        Number of augmented copies to emit per source episode (>= 1).
    seed:
        Base seed; each (variant, episode) pair gets a disjoint derived stream.
    include_original:
        When True, each source's un-augmented frames are emitted (stamped
        ``augmented=False``) before its variants.
    original_copies:
        How many copies of the original to emit per source when
        ``include_original`` is set (>= 1). Oversampling the originals
        reweights the training mix toward the nominal distribution, e.g.
        ``original_copies=2, variants=1`` yields a 67/33 original/augmented
        ratio.
    index_base:
        Global index of the first episode in ``episodes``. Lets a caller process
        sources one at a time (streaming) and still derive the same per-episode
        seeds as one in-memory call over the whole list.
    """
    eps = list(episodes)
    out: list[Episode] = []
    for i, ep in enumerate(eps):
        out.extend(
            iter_augment_episode(
                ep,
                pipeline_config,
                variants=variants,
                seed=seed,
                include_original=include_original,
                original_copies=original_copies,
                source_index=index_base + i,
            )
        )
    return TrainingSet(episodes=out)


def iter_augment_episode(
    episode: Episode,
    pipeline_config: list | None,
    *,
    variants: int = 1,
    seed: int | None = None,
    include_original: bool = False,
    original_copies: int = 1,
    source_index: int = 0,
) -> Iterator[Episode]:
    """Yield one source's original/variants one at a time.

    This is the bounded-memory form used by the CLI streaming writer. A
    dual-camera 720p episode can occupy several GiB uncompressed, so building all
    variants in a list can exhaust RAM even though each result is written
    immediately afterward.
    """
    if variants < 1:
        raise ValueError("variants must be >= 1")
    if original_copies < 1:
        raise ValueError("original_copies must be >= 1")
    steps = list(pipeline_config or [])
    if not steps:
        raise ValueError("no augmentations configured: the pipeline is empty")

    if include_original:
        for copy_index in range(original_copies):
            keep = episode.with_frames(episode.frames)
            keep.metadata["augmented"] = False
            if original_copies > 1:
                keep.metadata["original_copy"] = int(copy_index)
            yield keep
    for variant in range(variants):
        # A distinct index per (source, variant) so no two seasonings share an
        # RNG stream; global source index keeps streaming == in-memory.
        aug = _season(
            episode,
            steps,
            _derive_seed(seed, source_index * variants + variant),
        )
        aug.metadata["augmented"] = True
        if variants > 1:
            aug.metadata["variant"] = int(variant)
            aug.metadata["source_episode"] = int(source_index)
        yield aug


def sweep_episodes(
    episodes: Sequence[Episode],
    specs: Sequence[Mapping[str, Any]],
    *,
    seed: int | None = None,
    include_original: bool = False,
    index_base: int = 0,
) -> TrainingSet:
    """Deterministic magnitude sweep: apply each named single-effect pipeline in
    ``specs`` to every episode, producing one output per (episode, spec).

    Each spec is ``{"label": str, "pipeline": [step, ...]}`` where the pipeline is
    usually a single augmenter at a fixed magnitude (e.g. brightness +10%). Unlike
    :func:`augment_episodes` (which reseeds the whole pipeline), this isolates one
    effect at one magnitude per output, so the output grid is interpretable.

    ``index_base`` is the global index of the first episode, so a streaming caller
    that processes sources one at a time derives the same seeds as one bulk call.
    """
    eps = list(episodes)
    out: list[Episode] = []
    for i, ep in enumerate(eps):
        out.extend(
            iter_sweep_episode(
                ep,
                specs,
                seed=seed,
                include_original=include_original,
                source_index=index_base + i,
            )
        )
    return TrainingSet(episodes=out)


def iter_sweep_episode(
    episode: Episode,
    specs: Sequence[Mapping[str, Any]],
    *,
    seed: int | None = None,
    include_original: bool = False,
    source_index: int = 0,
) -> Iterator[Episode]:
    """Yield one source's sweep outputs one at a time."""
    steps = list(specs)
    if not steps:
        raise ValueError("no sweep steps configured")
    if include_original:
        keep = episode.with_frames(episode.frames)
        keep.metadata["augmented"] = False
        yield keep
    for step_index, spec in enumerate(steps):
        pipe = list(spec.get("pipeline") or [])
        if not pipe:
            raise ValueError(f"sweep step {spec.get('label', '?')!r} has an empty pipeline")
        aug = _season(
            episode,
            pipe,
            _derive_seed(seed, source_index * len(steps) + step_index),
        )
        aug.metadata["augmented"] = True
        if spec.get("label"):
            aug.metadata["sweep"] = spec["label"]
        aug.metadata["source_episode"] = int(source_index)
        yield aug


def _derive_seed(seed: int | None, index: int) -> int | None:
    if seed is None:
        return None
    # `seed + index` makes adjacent base seeds collide: run(S).episode[i+1] would
    # reuse run(S+1).episode[i]'s stream. Hash (seed, index) through SeedSequence
    # so distinct base seeds yield disjoint per-episode streams.
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _season(episode: Episode, pipeline_config: list, seed: int | None) -> Episode:
    """Run the pixel pipeline over one episode's frames, preserving provenance.

    Primary camera is seasoned first (params recorded in metadata). Every extra
    camera stream then replays that same history so dual-cam demos keep a
    consistent observation space (front + wrist, etc.).
    """
    if not pipeline_config:
        return episode
    pipeline = AugmentationPipeline.from_config(pipeline_config, seed=seed)
    frames, metadata = pipeline(episode.frames, episode.metadata)
    extras: dict[str, np.ndarray] = {}
    if episode.extra_videos:
        from lmfao.replay import replay_augmentation_history

        history = list(metadata.get("augmentation_history") or [])
        primary_shape = tuple(int(x) for x in episode.frames.shape)
        # Deterministic per-extra noise streams derived from the season seed.
        base = 0 if seed is None else int(seed)
        for i, (key, vid) in enumerate(sorted(episode.extra_videos.items())):
            extra_rng = np.random.default_rng(
                int(np.random.SeedSequence([base, 0xC0FFEE, i]).generate_state(1)[0])
            )
            extras[key] = replay_augmentation_history(
                vid,
                history,
                primary_shape=primary_shape,
                rng=extra_rng,
            )
    return episode.with_frames(frames, metadata=dict(metadata), extra_videos=extras)
