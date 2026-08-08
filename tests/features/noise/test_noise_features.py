from __future__ import annotations

import numpy as np
import pytest

from lmfao import AugmentationPipeline, build_augmenter, list_augmenters
from lmfao.features.noise import (
    CompressionArtifacts,
    DefocusBlur,
    GaussianNoise,
    ShotNoise,
    UniformNoise,
)
from lmfao.features.noise.accelerator import resolve_device
from lmfao.registry import get_augmenter

# The noises drawn per pixel, and the keyword that sets each one's strength.
NOISE = {"noise.gaussian": "sigma", "noise.uniform": "amplitude", "noise.shot": "strength"}

# Standard deviation each distribution produces per unit of strength. Shot noise
# is signal-dependent, so its entry is the value at the mid-gray `_video` below;
# the square-root law itself is checked separately.
SPREAD = {"noise.gaussian": 1.0, "noise.uniform": 1 / 12**0.5, "noise.shot": 0.5**0.5}


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=8):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def _detailed_video(frames=3, size=32):
    """Blur and compression need detail to destroy, so give them a busy frame."""
    return np.random.default_rng(0).integers(0, 256, (frames, size, size, 3), dtype=np.uint8)


def test_noise_augmenters_register_with_dotted_names():
    augmenters = list_augmenters()
    expected = {
        "noise.gaussian": GaussianNoise,
        "noise.uniform": UniformNoise,
        "noise.shot": ShotNoise,
        "noise.blur": DefocusBlur,
        "noise.compression": CompressionArtifacts,
    }

    for name, cls in expected.items():
        assert name in augmenters
        assert get_augmenter(name) is cls


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


@pytest.mark.parametrize("value", [16, 64, 144])
def test_shot_noise_grows_with_the_square_root_of_intensity(value):
    video = _video(value=value, size=32)
    augmenter = build_augmenter("noise.shot", strength=0.02, device="cpu")

    augmented, _ = augmenter(video, rng=np.random.default_rng(0))

    measured = (augmented.astype(np.float32) - video).std()
    assert measured == pytest.approx(0.02 * 255.0 * (value / 255.0) ** 0.5, rel=0.1)


# --- blur and compression: deterministic, and they read a pixel's neighbours ---


@pytest.mark.parametrize("name, params", [("noise.blur", {"radius": 1.5}), ("noise.compression", {"quality": 30.0})])
def test_neighbourhood_noise_preserves_shape_dtype_and_records_metadata(name, params):
    video = _detailed_video()
    augmenter = build_augmenter(name, **params)

    augmented, metadata = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert not np.array_equal(augmented, video)
    assert metadata["augmentation_params"][name] == params


@pytest.mark.parametrize("name, params", [("noise.blur", {"radius": 2.0}), ("noise.compression", {"quality": 10.0})])
def test_neighbourhood_noise_leaves_a_flat_frame_exactly_alone(name, params):
    """Nothing to smear or quantise, so any change here is a rounding or edge bug."""
    video = _video(value=128)

    augmented, _ = build_augmenter(name, **params)(video)

    np.testing.assert_array_equal(augmented, video)


@pytest.mark.parametrize("name, params", [("noise.blur", {"radius": 1.0}), ("noise.compression", {"quality": 30.0})])
def test_neighbourhood_noise_handles_sizes_that_are_not_multiples_of_eight(name, params):
    video = _detailed_video()[:, :23, :17, :]

    augmented, _ = build_augmenter(name, **params)(video)

    assert augmented.shape == video.shape


def test_blur_removes_detail_in_proportion_to_radius():
    video = _detailed_video()
    spreads = [build_augmenter("noise.blur", radius=r)(video)[0].std() for r in (0.5, 1.0, 2.0)]

    assert video.std() > spreads[0] > spreads[1] > spreads[2]


def test_lower_compression_quality_loses_more_detail():
    video = _detailed_video()
    errors = [
        np.abs(build_augmenter("noise.compression", quality=q)(video)[0].astype(np.float32) - video).mean()
        for q in (95.0, 60.0, 20.0)
    ]

    assert errors[0] < errors[1] < errors[2]
    assert errors[0] < 2.0  # near-lossless at the top of the scale


def test_noise_pipeline_is_reproducible_with_seed():
    video = _detailed_video()
    pipeline = AugmentationPipeline.from_config(
        [
            {"name": "noise.gaussian", "params": {}},
            {"name": "noise.uniform", "params": {}},
            {"name": "noise.shot", "params": {}},
            {"name": "noise.blur", "params": {}},
            {"name": "noise.compression", "params": {}},
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

    with pytest.raises(ValueError, match="strength must be"):
        build_augmenter("noise.shot", strength=-0.1)

    with pytest.raises(ValueError, match="radius must be"):
        build_augmenter("noise.blur", radius=0)

    with pytest.raises(ValueError, match="quality must be"):
        build_augmenter("noise.compression", quality=101)

    with pytest.raises(ValueError, match="unknown device"):
        build_augmenter("noise.gaussian", device="tpu")(_video())
