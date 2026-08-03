import numpy as np
import pytest
from conftest import PUCK_COLOR, build_episode

from lmfao import generate_training_set
from lmfao.datasets import Episode
from lmfao.program import TrainingSet, augment_episodes, sweep_episodes

_PIPE = [
    {"name": "lighting.brightness", "params": {"factor": 0.5}},
    {"name": "noise.gaussian", "params": {"sigma": 0.02}},
]


def _plain(n=2):
    return [Episode(frames=np.full((4, 8, 8, 3), 100, np.uint8), state=np.zeros((4, 3)),
                    fps=30.0, task="t", metadata={"episode_index": i}) for i in range(n)]


def test_augment_variants_and_originals_counts():
    ts = augment_episodes(_plain(2), _PIPE, variants=3, seed=5, include_original=True)
    assert len(ts.episodes) == 8  # 2 originals + 2*3 variants
    aug = [e for e in ts.episodes if e.metadata.get("augmented")]
    orig = [e for e in ts.episodes if not e.metadata.get("augmented")]
    assert len(aug) == 6 and len(orig) == 2


def test_augment_applies_pipeline_and_preserves_state():
    ts = augment_episodes(_plain(1), _PIPE, variants=1, seed=1)
    ep = ts.episodes[0]
    assert abs(float(ep.frames.mean()) - 50) < 4  # brightness 0.5 of 100
    assert ep.state is not None and ep.state.shape == (4, 3)
    assert [h["name"] for h in ep.metadata["augmentation_history"]] == \
        ["lighting.brightness", "noise.gaussian"]


def test_augment_variants_are_independent_but_reproducible():
    a = augment_episodes(_plain(1), _PIPE, variants=2, seed=5)
    b = augment_episodes(_plain(1), _PIPE, variants=2, seed=5)
    assert not np.array_equal(a.episodes[0].frames, a.episodes[1].frames)  # variants differ
    assert np.array_equal(a.episodes[0].frames, b.episodes[0].frames)  # reproducible


def test_augment_does_not_mutate_source_episodes():
    src = _plain(1)
    before = src[0].frames.copy()
    augment_episodes(src, _PIPE, variants=2, seed=1)
    assert np.array_equal(src[0].frames, before)


def test_augment_rejects_empty_pipeline_and_bad_variants():
    with pytest.raises(ValueError, match="empty"):
        augment_episodes(_plain(1), [], variants=1)
    with pytest.raises(ValueError, match="variants"):
        augment_episodes(_plain(1), _PIPE, variants=0)


def _bright(tag, factor):
    return {"label": f"brightness {tag}",
            "pipeline": [{"name": "lighting.brightness", "params": {"factor": factor}}]}


def test_sweep_one_output_per_episode_per_spec():
    specs = [_bright("-10%", 0.9), _bright("+10%", 1.1), _bright("+20%", 1.2)]
    ts = sweep_episodes(_plain(2), specs, seed=1)
    assert len(ts.episodes) == 6  # 2 episodes x 3 specs
    labels = {e.metadata["sweep"] for e in ts.episodes}
    assert labels == {"brightness -10%", "brightness +10%", "brightness +20%"}
    assert all(e.metadata.get("augmented") for e in ts.episodes)


def test_sweep_applies_the_specified_magnitude():
    specs = [_bright("-20%", 0.8), _bright("+20%", 1.2)]
    ts = sweep_episodes(_plain(1), specs, seed=1)
    by = {e.metadata["sweep"]: e.frames.mean() for e in ts.episodes}
    assert by["brightness -20%"] < 100 < by["brightness +20%"]  # 100 is the source value


def test_sweep_includes_originals_and_is_reproducible():
    specs = [_bright("+10%", 1.1)]
    a = sweep_episodes(_plain(1), specs, seed=3, include_original=True)
    b = sweep_episodes(_plain(1), specs, seed=3, include_original=True)
    assert sum(1 for e in a.episodes if not e.metadata.get("augmented")) == 1
    assert all(np.array_equal(x.frames, y.frames) for x, y in zip(a.episodes, b.episodes))


def test_sweep_rejects_empty():
    with pytest.raises(ValueError, match="no sweep steps"):
        sweep_episodes(_plain(1), [])


def _reals(n=3):
    return [build_episode(seed=s) for s in range(n)]


def _config(enabled=True, n_synthetic=4):
    return {
        "miniworld": {
            "enabled": enabled,
            "n_synthetic": n_synthetic,
            "puck_color": PUCK_COLOR,
            "object_pose_region": ((-0.05, -0.05), (0.05, 0.05)),
            "seed": 7,
        },
        "pipeline": [
            {"name": "lighting.color_temperature", "params": {"intensity": 0.35}, "probability": 1.0},
        ],
    }


def test_switch_off_is_pure_v1():
    reals = _reals(3)
    ts = generate_training_set(reals, _config(enabled=False))
    assert isinstance(ts, TrainingSet)
    assert ts.num_synthetic == 0
    assert ts.num_real == 3
    assert ts.summary()["episodes"] == 3


def test_switch_on_merges_and_seasons():
    reals = _reals(3)
    ts = generate_training_set(reals, _config(enabled=True, n_synthetic=4), seed=42)
    assert ts.num_real == 3
    assert ts.num_synthetic == 4
    assert len(ts.episodes) == 7
    # every episode, real or synthetic, was seasoned by the pixel pipeline
    for ep in ts.episodes:
        assert ep.metadata.get("augmentations") == ["lighting.color_temperature"]


def test_missing_miniworld_block_defaults_to_real_only():
    reals = _reals(2)
    ts = generate_training_set(reals, {"pipeline": []})
    assert ts.num_synthetic == 0
    assert ts.num_real == 2


def test_provenance_survives_pipeline():
    reals = _reals(2)
    ts = generate_training_set(reals, _config(n_synthetic=2), seed=1)
    synth = [e for e in ts.episodes if e.is_synthetic]
    assert len(synth) == 2
    for ep in synth:
        # provenance stamped by the generator is still present after seasoning
        assert ep.metadata["synthetic"] is True
        assert "miniworld" in ep.metadata
        assert "augmentations" in ep.metadata  # and pipeline info was added


def test_reproducible_with_seed():
    reals = _reals(2)
    a = generate_training_set(reals, _config(n_synthetic=3), seed=123)
    b = generate_training_set(_reals(2), _config(n_synthetic=3), seed=123)
    assert a.summary() == b.summary()
    for ea, eb in zip(a.episodes, b.episodes):
        np.testing.assert_array_equal(ea.frames, eb.frames)


def test_empty_pipeline_still_generates():
    reals = _reals(2)
    cfg = _config(n_synthetic=2)
    cfg["pipeline"] = []
    ts = generate_training_set(reals, cfg, seed=5)
    assert ts.num_synthetic == 2
    # with no pipeline, synthetic frames are the raw render (no augmentations key)
    synth = [e for e in ts.episodes if e.is_synthetic][0]
    assert "augmentations" not in synth.metadata


def test_frames_tensor_stacks_all_episodes():
    reals = _reals(2)
    ts = generate_training_set(reals, _config(n_synthetic=2), seed=2)
    frames = ts.frames()
    assert frames.shape[0] == ts.num_frames
    assert frames.shape[1:] == reals[0].frames.shape[1:]
