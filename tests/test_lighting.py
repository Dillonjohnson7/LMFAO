import numpy as np

from lmfao import AugmentationPipeline
from lmfao.features.lighting import ColorTemperatureShift
from lmfao.registry import get_augmenter


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=8):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def test_registers_under_expected_name():
    assert get_augmenter("lighting.color_temperature") is ColorTemperatureShift


def test_preserves_shape_and_dtype():
    video = _video()
    augmenter = ColorTemperatureShift(shift=-1.0)

    augmented, _ = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype


def test_red_shift_boosts_red_and_cuts_blue():
    video = _video(value=128)
    augmenter = ColorTemperatureShift(shift=-1.0, intensity=0.5)

    augmented, _ = augmenter(video)

    assert augmented[..., 0].mean() > video[..., 0].mean()
    assert augmented[..., 2].mean() < video[..., 2].mean()


def test_blue_shift_boosts_blue_and_cuts_red():
    video = _video(value=128)
    augmenter = ColorTemperatureShift(shift=1.0, intensity=0.5)

    augmented, _ = augmenter(video)

    assert augmented[..., 2].mean() > video[..., 2].mean()
    assert augmented[..., 0].mean() < video[..., 0].mean()


def test_zero_shift_is_a_no_op():
    video = _video(value=128)
    augmenter = ColorTemperatureShift(shift=0.0)

    augmented, _ = augmenter(video)

    np.testing.assert_array_equal(augmented, video)


def test_records_metadata():
    video = _video()
    augmenter = ColorTemperatureShift(shift=-1.0)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"][augmenter.name]
    assert params["shift"] == -1.0
    assert params["red_gain"] > 1.0
    assert params["blue_gain"] < 1.0
    assert params["applied_to_color_channels"] is True


def test_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "lighting.color_temperature", "params": {}}],
        seed=99,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_clips_instead_of_overflowing_uint8():
    video = _video(value=250)
    augmenter = ColorTemperatureShift(shift=-1.0, intensity=1.0)

    augmented, _ = augmenter(video)

    assert augmented.max() <= 255
    assert augmented.dtype == np.uint8


def test_grayscale_video_is_passed_through_unaffected():
    video = _video(channels=1, value=128)
    augmenter = ColorTemperatureShift(shift=-1.0)

    augmented, metadata = augmenter(video)

    np.testing.assert_array_equal(augmented, video)
    assert metadata["augmentation_params"][augmenter.name]["applied_to_color_channels"] is False


def test_invalid_shift_range_raises():
    video = _video()
    augmenter = ColorTemperatureShift(min_shift=0.5, max_shift=-0.5)

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for min_shift > max_shift")
