import numpy as np

from lmfao import AugmentationPipeline
from lmfao.features.spatial import RandomCrop
from lmfao.registry import get_augmenter


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=24):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def test_registers_under_expected_name():
    assert get_augmenter("spatial.random_crop") is RandomCrop


def test_preserves_shape_and_dtype():
    video = _video()
    augmenter = RandomCrop(pad=4, shift_x=2, shift_y=-1)

    augmented, _ = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype


def test_zero_shift_is_identity():
    video = _video()
    augmenter = RandomCrop(pad=4, shift_x=0, shift_y=0, pad_mode="reflect")

    augmented, _ = augmenter(video)

    np.testing.assert_array_equal(augmented, video)


def test_explicit_shift_moves_content_by_expected_offset_and_applies_to_every_frame():
    frames, size, channels = 3, 20, 3
    video = np.zeros((frames, size, size, channels), dtype=np.uint8)
    marker_row, marker_col = 10, 10
    video[:, marker_row, marker_col, 0] = 255

    augmenter = RandomCrop(pad=5, shift_x=3, shift_y=-2, pad_mode="zero")
    augmented, _ = augmenter(video)

    expected_row = marker_row - (-2)
    expected_col = marker_col - 3
    for f in range(frames):
        assert augmented[f, expected_row, expected_col, 0] == 255
        assert int(augmented[f].sum()) == 255


def test_records_metadata():
    video = _video()
    augmenter = RandomCrop(pad=4, shift_x=2, shift_y=-1, pad_mode="zero")

    _, metadata = augmenter(video)

    assert metadata["augmentation_params"][augmenter.name] == {
        "shift_x": [2, 2, 2, 2],
        "shift_y": [-1, -1, -1, -1],
        "pad": 4,
        "pad_mode": "zero",
    }


def test_explicit_shift_is_clamped_to_pad_range():
    video = _video()
    augmenter = RandomCrop(pad=3, shift_x=100, shift_y=-100)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"][augmenter.name]
    assert params["shift_x"] == [3, 3, 3, 3]
    assert params["shift_y"] == [-3, -3, -3, -3]


def test_random_shift_is_bounded_by_pad():
    video = _video()
    for _ in range(20):
        augmenter = RandomCrop(pad=4)
        _, metadata = augmenter(video)
        params = metadata["augmentation_params"][augmenter.name]
        assert all(-4 <= v <= 4 for v in params["shift_x"])
        assert all(-4 <= v <= 4 for v in params["shift_y"])


def test_random_shift_is_sampled_independently_per_frame():
    video = _video(frames=50)
    augmenter = RandomCrop(pad=4)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"][augmenter.name]
    assert len(set(params["shift_x"])) > 1
    assert len(set(params["shift_y"])) > 1


def test_random_shift_actually_moves_each_frame_independently():
    frames, size, channels = 30, 20, 3
    video = np.zeros((frames, size, size, channels), dtype=np.uint8)
    marker_row, marker_col = 10, 10
    video[:, marker_row, marker_col, 0] = 255

    augmenter = RandomCrop(pad=6, pad_mode="zero")
    augmented, _ = augmenter(video)

    marker_positions = set()
    for f in range(frames):
        positions = list(zip(*np.nonzero(augmented[f, ..., 0])))
        assert len(positions) == 1
        marker_positions.add(positions[0])

    assert len(marker_positions) > 1


def test_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "spatial.random_crop", "params": {"pad": 4}}],
        seed=71,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_pad_larger_than_frame_raises():
    video = _video(size=8)
    augmenter = RandomCrop(pad=8)

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when pad >= frame dimension")


def test_invalid_pad_mode_raises():
    video = _video()
    augmenter = RandomCrop(pad_mode="blur")

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid pad_mode")


def test_non_positive_pad_raises():
    video = _video()
    augmenter = RandomCrop(pad=0)

    try:
        augmenter(video)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive pad")
