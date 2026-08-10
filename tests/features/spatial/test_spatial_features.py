from __future__ import annotations

import numpy as np
import pytest

from lmfao import AugmentationPipeline, build_augmenter, list_augmenters
from lmfao.features.spatial import RandomCrop
from lmfao.registry import get_augmenter


def _video(value=128, channels=3, dtype=np.uint8, frames=4, size=24):
    return np.full((frames, size, size, channels), value, dtype=dtype)


def test_spatial_augmenters_register_with_dotted_names():
    assert "spatial.random_crop" in list_augmenters()
    assert get_augmenter("spatial.random_crop") is RandomCrop


def test_random_crop_preserves_shape_and_dtype():
    video = _video()
    augmenter = build_augmenter("spatial.random_crop", pad=4, shift_x=2, shift_y=-1)

    augmented, _ = augmenter(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype


def test_random_crop_zero_shift_is_identity():
    video = _video()
    augmenter = build_augmenter("spatial.random_crop", pad=4, shift_x=0, shift_y=0, pad_mode="reflect")

    augmented, _ = augmenter(video)

    np.testing.assert_array_equal(augmented, video)


def test_random_crop_explicit_shift_moves_content_by_expected_offset():
    frames, size, channels = 3, 20, 3
    video = np.zeros((frames, size, size, channels), dtype=np.uint8)
    marker_row, marker_col = 10, 10
    video[:, marker_row, marker_col, 0] = 255

    augmenter = build_augmenter("spatial.random_crop", pad=5, shift_x=3, shift_y=-2, pad_mode="zero")
    augmented, _ = augmenter(video)

    expected_row = marker_row - (-2)
    expected_col = marker_col - 3
    for frame_index in range(frames):
        assert augmented[frame_index, expected_row, expected_col, 0] == 255
        assert int(augmented[frame_index].sum()) == 255


def test_random_crop_records_metadata():
    video = _video()
    augmenter = build_augmenter("spatial.random_crop", pad=4, shift_x=2, shift_y=-1, pad_mode="zero")

    _, metadata = augmenter(video)

    assert metadata["augmentation_params"]["spatial.random_crop"] == {
        "shift_x": [2, 2, 2, 2],
        "shift_y": [-1, -1, -1, -1],
        "pad": 4,
        "pad_mode": "zero",
        "temporal_mode": "per_frame",
    }


def test_random_crop_explicit_shift_is_clamped_to_pad_range():
    video = _video()
    augmenter = build_augmenter("spatial.random_crop", pad=3, shift_x=100, shift_y=-100)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"]["spatial.random_crop"]
    assert params["shift_x"] == [3, 3, 3, 3]
    assert params["shift_y"] == [-3, -3, -3, -3]


def test_random_crop_random_shift_is_sampled_independently_per_frame():
    video = _video(frames=50)
    augmenter = build_augmenter("spatial.random_crop", pad=4)

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"]["spatial.random_crop"]
    assert len(set(params["shift_x"])) > 1
    assert len(set(params["shift_y"])) > 1
    assert all(-4 <= value <= 4 for value in params["shift_x"])
    assert all(-4 <= value <= 4 for value in params["shift_y"])


def test_random_crop_random_shift_moves_each_frame_independently():
    frames, size, channels = 30, 20, 3
    video = np.zeros((frames, size, size, channels), dtype=np.uint8)
    marker_row, marker_col = 10, 10
    video[:, marker_row, marker_col, 0] = 255

    augmenter = build_augmenter("spatial.random_crop", pad=6, pad_mode="zero")
    augmented, _ = augmenter(video)

    marker_positions = set()
    for frame_index in range(frames):
        positions = list(zip(*np.nonzero(augmented[frame_index, ..., 0])))
        assert len(positions) == 1
        marker_positions.add(positions[0])

    assert len(marker_positions) > 1


def test_random_crop_pipeline_is_reproducible_with_seed():
    video = _video()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "spatial.random_crop", "params": {"pad": 4}}],
        seed=71,
    )

    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_random_crop_config_validation():
    with pytest.raises(ValueError, match="pad must be > 0"):
        build_augmenter("spatial.random_crop", pad=0)

    with pytest.raises(ValueError, match="pad_mode must be one of"):
        build_augmenter("spatial.random_crop", pad_mode="blur")

    with pytest.raises(ValueError, match="temporal_mode must be"):
        build_augmenter("spatial.random_crop", temporal_mode="weekly")

    augmenter = build_augmenter("spatial.random_crop", pad=8)
    with pytest.raises(ValueError, match="must be smaller than both height"):
        augmenter(_video(size=8))


def test_random_crop_constant_mode_holds_one_shift_for_whole_episode():
    video = _video(frames=50)
    augmenter = build_augmenter("spatial.random_crop", pad=4, temporal_mode="constant")

    _, metadata = augmenter(video)

    params = metadata["augmentation_params"]["spatial.random_crop"]
    assert params["temporal_mode"] == "constant"
    assert len(set(params["shift_x"])) == 1
    assert len(set(params["shift_y"])) == 1
    assert all(-4 <= value <= 4 for value in params["shift_x"])
    assert all(-4 <= value <= 4 for value in params["shift_y"])


def test_random_crop_constant_mode_moves_every_frame_identically():
    frames, size, channels = 12, 20, 3
    video = np.zeros((frames, size, size, channels), dtype=np.uint8)
    video[:, 10, 10, 0] = 255

    augmenter = build_augmenter("spatial.random_crop", pad=6, pad_mode="zero", temporal_mode="constant")
    augmented, _ = augmenter(video)

    positions = set()
    for frame_index in range(frames):
        frame_positions = list(zip(*np.nonzero(augmented[frame_index, ..., 0])))
        assert len(frame_positions) == 1
        positions.add(frame_positions[0])
    assert len(positions) == 1


def test_random_crop_constant_mode_is_reproducible_with_seed():
    video = _video(frames=20)
    pipeline = AugmentationPipeline.from_config(
        [{"name": "spatial.random_crop", "params": {"pad": 4, "temporal_mode": "constant"}}],
        seed=71,
    )

    first, _ = pipeline(video)
    second, _ = pipeline(video)

    np.testing.assert_array_equal(first, second)
