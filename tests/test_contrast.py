import numpy as np

from lmfao import AugmentationPipeline
from lmfao.features.lighting import ContrastScale
from lmfao.registry import get_augmenter


def _gradient_video(channels=3, dtype=np.uint8, frames=4, size=16):
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    frame = np.tile(ramp, (size, 1))
    frame = np.stack([frame] * channels, axis=-1).astype(dtype)
    return np.stack([frame] * frames)


def test_registers_under_expected_name():
    assert get_augmenter("lighting.contrast") is ContrastScale


def test_preserves_shape_and_dtype():
    video = _gradient_video()
    augmenter = ContrastScale(factor=1.3)

    augmented, _ = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype


def test_factor_above_one_increases_spread():
    video = _gradient_video()
    augmenter = ContrastScale(factor=1.5)

    augmented, _ = augmenter(video)

    assert augmented.astype(np.float32).std() > video.astype(np.float32).std()


def test_factor_below_one_decreases_spread():
    video = _gradient_video()
    augmenter = ContrastScale(factor=0.5)

    augmented, _ = augmenter(video)

    assert augmented.astype(np.float32).std() < video.astype(np.float32).std()


def test_factor_of_one_is_a_no_op():
    video = _gradient_video()
    augmenter = ContrastScale(factor=1.0)

    augmented, _ = augmenter(video)

    np.testing.assert_array_equal(augmented, video)


def test_pivots_around_mid_gray_for_uint8():
    video = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
    augmenter = ContrastScale(factor=2.0)

    augmented, metadata = augmenter(video)

    np.testing.assert_array_equal(augmented, video)
    assert metadata["augmentation_params"][augmenter.name]["pivot"] == 127.5


def test_records_metadata():
    video = _gradient_video()
    augmenter = ContrastScale(factor=1.3)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"][augmenter.name]
    assert params["factor"] == 1.3
    assert params["pivot"] == 127.5


def test_pipeline_is_reproducible_with_seed():
    video = _gradient_video()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "lighting.contrast", "params": {}}],
        seed=31,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_clips_instead_of_overflowing_uint8():
    video = _gradient_video()
    augmenter = ContrastScale(factor=3.0)

    augmented, _ = augmenter(video)

    assert augmented.max() <= 255
    assert augmented.min() >= 0
    assert augmented.dtype == np.uint8


def test_invalid_factor_range_raises():
    video = _gradient_video()
    augmenter = ContrastScale(min_factor=1.5, max_factor=0.5)

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for min_factor > max_factor")
