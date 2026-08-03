from __future__ import annotations

import numpy as np
import pytest

from lmfao import AugmentationPipeline, build_augmenter, list_augmenters
from lmfao.features.lighting import BrightnessScale, ColorTemperatureShift, ContrastScale
from lmfao.registry import get_augmenter


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=8):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def _gradient_video(channels=3, dtype=np.uint8, frames=4, size=16):
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    frame = np.tile(ramp, (size, 1))
    frame = np.stack([frame] * channels, axis=-1).astype(dtype)
    return np.stack([frame] * frames)


def test_lighting_augmenters_register_with_dotted_names():
    augmenters = list_augmenters()

    assert "lighting.brightness" in augmenters
    assert "lighting.color_temperature" in augmenters
    assert "lighting.contrast" in augmenters
    assert get_augmenter("lighting.brightness") is BrightnessScale
    assert get_augmenter("lighting.color_temperature") is ColorTemperatureShift
    assert get_augmenter("lighting.contrast") is ContrastScale


def test_brightness_scales_intensity_and_records_metadata():
    video = _video(value=100)
    augmenter = build_augmenter("lighting.brightness", factor=1.3)

    augmented, metadata = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert augmented.mean() > video.mean()
    assert metadata["augmentation_params"]["lighting.brightness"] == {"factor": 1.3}


def test_brightness_clips_uint8_bounds():
    video = _video(value=240)
    augmenter = build_augmenter("lighting.brightness", factor=1.4)

    augmented, _ = augmenter(video)

    assert augmented.max() <= 255
    assert augmented.dtype == np.uint8


def test_contrast_scales_around_mid_gray_and_records_metadata():
    video = _gradient_video()
    augmenter = build_augmenter("lighting.contrast", factor=1.5)

    augmented, metadata = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert augmented.astype(np.float32).std() > video.astype(np.float32).std()
    assert metadata["augmentation_params"]["lighting.contrast"]["factor"] == 1.5
    assert metadata["augmentation_params"]["lighting.contrast"]["pivot"] == 127.5


def test_contrast_factor_below_one_decreases_spread():
    video = _gradient_video()
    augmenter = build_augmenter("lighting.contrast", factor=0.5)

    augmented, _ = augmenter(video)

    assert augmented.astype(np.float32).std() < video.astype(np.float32).std()


def test_color_temperature_shifts_channel_balance_and_records_metadata():
    video = _video(value=128)
    augmenter = build_augmenter("lighting.color_temperature", shift=-1.0, intensity=0.5)

    augmented, metadata = augmenter(video)
    params = metadata["augmentation_params"]["lighting.color_temperature"]

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert augmented[..., 0].mean() > video[..., 0].mean()
    assert augmented[..., 2].mean() < video[..., 2].mean()
    assert params["shift"] == -1.0
    assert params["red_gain"] > 1.0
    assert params["blue_gain"] < 1.0
    assert params["applied_to_color_channels"] is True


def test_color_temperature_grayscale_video_is_passed_through_unaffected():
    video = _video(channels=1, value=128)
    augmenter = build_augmenter("lighting.color_temperature", shift=-1.0)

    augmented, metadata = augmenter(video)

    np.testing.assert_array_equal(augmented, video)
    assert metadata["augmentation_params"]["lighting.color_temperature"]["applied_to_color_channels"] is False


def test_lighting_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [
            {"name": "lighting.brightness", "params": {}},
            {"name": "lighting.color_temperature", "params": {}},
            {"name": "lighting.contrast", "params": {}},
        ],
        seed=99,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_lighting_config_validation():
    with pytest.raises(ValueError, match="min_factor/max_factor"):
        build_augmenter("lighting.brightness", min_factor=1.5, max_factor=0.5)

    with pytest.raises(ValueError, match="min_factor/max_factor"):
        build_augmenter("lighting.contrast", min_factor=1.5, max_factor=0.5)

    with pytest.raises(ValueError, match="min_shift/max_shift"):
        build_augmenter("lighting.color_temperature", min_shift=0.5, max_shift=-0.5)


# --- LUT fast path: integer video must be bit-identical to the float reference ---

def _ref_pointwise(video, transform):
    from lmfao.base import preserve_dtype
    return preserve_dtype(video, transform(video.astype(np.float32)))


@pytest.mark.parametrize("factor", [0.5, 0.85, 0.95, 1.05, 1.15, 1.5, 2.0])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_brightness_lut_matches_float_path(factor, channels):
    rng = np.random.default_rng(0)
    v = rng.integers(0, 256, (3, 12, 16, channels), dtype=np.uint8)
    out, _ = BrightnessScale(factor=factor)(v, {}, np.random.default_rng(1))
    assert np.array_equal(out, _ref_pointwise(v, lambda x: x * factor))


@pytest.mark.parametrize("factor", [0.6, 0.9, 1.1, 1.6])
def test_contrast_lut_matches_float_path(factor):
    rng = np.random.default_rng(0)
    v = rng.integers(0, 256, (3, 12, 16, 3), dtype=np.uint8)
    out, _ = ContrastScale(factor=factor)(v, {}, np.random.default_rng(1))
    assert np.array_equal(out, _ref_pointwise(v, lambda x: (x - 127.5) * factor + 127.5))


@pytest.mark.parametrize("shift", [-0.6, -0.2, 0.2, 0.6])
def test_color_temperature_lut_matches_float_path(shift):
    rng = np.random.default_rng(0)
    v = rng.integers(0, 256, (3, 12, 16, 3), dtype=np.uint8)
    out, m = ColorTemperatureShift(shift=shift, intensity=0.35)(v, {}, np.random.default_rng(1))
    p = m["augmentation_params"]["lighting.color_temperature"]
    ref = v.astype(np.float32)
    ref[..., 0] *= p["red_gain"]
    ref[..., 2] *= p["blue_gain"]
    from lmfao.base import preserve_dtype
    assert np.array_equal(out, preserve_dtype(v, ref))
