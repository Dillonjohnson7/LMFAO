import numpy as np

from lmfao import AugmentationPipeline
from lmfao.features.lighting import BrightnessScale
from lmfao.registry import get_augmenter


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=8):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def test_registers_under_expected_name():
    assert get_augmenter("lighting.brightness") is BrightnessScale


def test_preserves_shape_and_dtype():
    video = _video()
    augmenter = BrightnessScale(factor=1.2)

    augmented, _ = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype


def test_factor_above_one_brightens():
    video = _video(value=100)
    augmenter = BrightnessScale(factor=1.3)

    augmented, _ = augmenter(video)

    assert augmented.mean() > video.mean()


def test_factor_below_one_darkens():
    video = _video(value=100)
    augmenter = BrightnessScale(factor=0.7)

    augmented, _ = augmenter(video)

    assert augmented.mean() < video.mean()


def test_factor_of_one_is_a_no_op():
    video = _video(value=100)
    augmenter = BrightnessScale(factor=1.0)

    augmented, _ = augmenter(video)

    np.testing.assert_array_equal(augmented, video)


def test_records_metadata():
    video = _video()
    augmenter = BrightnessScale(factor=1.3)

    _, metadata = augmenter(video)

    assert metadata["augmentation_params"][augmenter.name] == {"factor": 1.3}


def test_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "lighting.brightness", "params": {}}],
        seed=17,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_clips_instead_of_overflowing_uint8():
    video = _video(value=240)
    augmenter = BrightnessScale(factor=1.4)

    augmented, _ = augmenter(video)

    assert augmented.max() <= 255
    assert augmented.dtype == np.uint8


def test_clips_instead_of_underflowing_uint8():
    video = _video(value=10)
    augmenter = BrightnessScale(factor=0.1)

    augmented, _ = augmenter(video)

    assert augmented.min() >= 0
    assert augmented.dtype == np.uint8


def test_invalid_factor_range_raises():
    video = _video()
    augmenter = BrightnessScale(min_factor=1.5, max_factor=0.5)

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for min_factor > max_factor")
