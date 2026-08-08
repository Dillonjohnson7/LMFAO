from __future__ import annotations

import numpy as np
import pytest

from lmfao import AugmentationPipeline, build_augmenter, list_augmenters
from lmfao.features.noise import GaussianNoise, UniformNoise
from lmfao.features.noise.accelerator import resolve_device
from lmfao.registry import get_augmenter

NOISE = {"noise.gaussian": "sigma", "noise.uniform": "amplitude"}

# Standard deviation each distribution produces per unit of strength.
SPREAD = {"noise.gaussian": 1.0, "noise.uniform": 1 / 12**0.5}


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=8):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def test_noise_augmenters_register_with_dotted_names():
    augmenters = list_augmenters()

    assert "noise.gaussian" in augmenters
    assert "noise.uniform" in augmenters
    assert get_augmenter("noise.gaussian") is GaussianNoise
    assert get_augmenter("noise.uniform") is UniformNoise


@pytest.mark.parametrize("name, strength", NOISE.items())
def test_noise_preserves_shape_dtype_and_records_metadata(name, strength):
    video = _video()
    augmenter = build_augmenter(name, device="cpu", **{strength: 0.05})

    augmented, metadata = augmenter(video, rng=np.random.default_rng(0))

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert not np.array_equal(augmented, video)
    assert metadata["augmentation_params"][name] == {strength: 0.05, "device": "cpu"}


@pytest.mark.parametrize("name, strength", NOISE.items())
def test_zero_strength_leaves_the_video_untouched(name, strength):
    video = _video()
    augmenter = build_augmenter(name, device="cpu", **{strength: 0.0})

    augmented, _ = augmenter(video, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(augmented, video)


@pytest.mark.parametrize("name", NOISE)
def test_noise_is_resampled_for_every_frame(name):
    augmented, _ = build_augmenter(name, device="cpu")(_video(), rng=np.random.default_rng(0))

    assert not np.array_equal(augmented[0], augmented[1])


@pytest.mark.parametrize("dtype, value, dynamic_range", [(np.uint8, 128, 255.0), (np.float32, 0.5, 1.0)])
@pytest.mark.parametrize("name, strength", NOISE.items())
def test_strength_is_a_fraction_of_the_dtype_range(dtype, value, dynamic_range, name, strength):
    video = _video(value=value, dtype=dtype, size=32)
    augmenter = build_augmenter(name, device="cpu", **{strength: 0.05})

    augmented, _ = augmenter(video, rng=np.random.default_rng(0))

    measured = (augmented.astype(np.float32) - video).std()
    assert measured == pytest.approx(0.05 * SPREAD[name] * dynamic_range, rel=0.1)


def test_integer_video_rounds_and_clips_like_the_float_reference():
    video = _video(value=250)
    augmenter = build_augmenter("noise.gaussian", sigma=0.5, device="cpu")

    augmented, _ = augmenter(video, rng=np.random.default_rng(0))

    noise = np.random.default_rng(0).standard_normal(video.shape, dtype=np.float32) * (0.5 * 255.0)
    assert np.array_equal(augmented, np.clip(np.rint(noise + video), 0, 255).astype(np.uint8))


def test_noise_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [
            {"name": "noise.gaussian", "params": {}},
            {"name": "noise.uniform", "params": {}},
        ],
        seed=99,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_auto_device_is_resolved_and_recorded():
    video = _video()

    _, metadata = build_augmenter("noise.gaussian")(video, rng=np.random.default_rng(0))

    assert metadata["augmentation_params"]["noise.gaussian"]["device"] == resolve_device("auto", video)


def test_dtypes_no_accelerator_can_hold_fall_back_to_cpu():
    video = _video(value=0.5, dtype=np.float64)

    assert resolve_device("auto", video) == "cpu"
    with pytest.raises(ValueError, match="cannot sample for dtype"):
        resolve_device("mps", video)


def test_noise_config_validation():
    with pytest.raises(ValueError, match="sigma must be"):
        build_augmenter("noise.gaussian", sigma=-0.1)

    with pytest.raises(ValueError, match="amplitude must be"):
        build_augmenter("noise.uniform", amplitude=float("nan"))

    with pytest.raises(ValueError, match="unknown device"):
        build_augmenter("noise.gaussian", device="tpu")(_video())
