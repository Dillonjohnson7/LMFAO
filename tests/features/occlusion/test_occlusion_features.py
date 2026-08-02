from __future__ import annotations

import numpy as np
import pytest

from lmfao import AugmentationPipeline, build_augmenter, list_augmenters


def test_occlusion_augmenters_register_with_dotted_names():
    augmenters = list_augmenters()

    assert "occlusion.sequence_box" in augmenters
    assert "occlusion.border_intrusion" in augmenters
    assert "occlusion.moving_box" in augmenters


def test_sequence_box_masks_same_region_across_frames_and_records_metadata():
    video = np.full((3, 12, 12, 3), 100, dtype=np.uint8)
    pipeline = AugmentationPipeline.from_config(
        [
            {
                "name": "occlusion.sequence_box",
                "params": {
                    "box_area_range": (0.20, 0.20),
                    "aspect_ratio_range": (1.0, 1.0),
                },
            }
        ],
        seed=7,
    )

    augmented, metadata = pipeline(video)
    box = metadata["augmentation_params"]["occlusion.sequence_box"]["box"]
    top = box["top"]
    left = box["left"]
    bottom = top + box["height"]
    right = left + box["width"]

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert np.all(augmented[:, top:bottom, left:right, :] == 0)
    assert np.all(augmented[:, :top, :, :] == 100)
    assert metadata["augmentation_params"]["occlusion.sequence_box"]["temporal_mode"] == "sequence_constant"


def test_border_intrusion_masks_configured_edge():
    video = np.full((2, 10, 10, 3), 100, dtype=np.uint8)
    augmenter = build_augmenter(
        "occlusion.border_intrusion",
        edges=("bottom",),
        max_fraction=0.20,
    )

    augmented, metadata = augmenter(video, rng=np.random.default_rng(4))
    region = metadata["augmentation_params"]["occlusion.border_intrusion"]["regions"][0]
    top = region["top"]

    assert region["edge"] == "bottom"
    assert region["height"] >= 1
    assert np.all(augmented[:, top:, :, :] == 0)
    assert np.all(augmented[:, :top, :, :] == 100)


def test_moving_box_records_smooth_positions_and_preserves_video_contract():
    video = np.full((4, 16, 16, 3), 100, dtype=np.uint8)
    augmenter = build_augmenter(
        "occlusion.moving_box",
        box_area_range=(0.10, 0.10),
        aspect_ratio_range=(1.0, 1.0),
        velocity_range=(1.0, 1.0),
        edge_bounce=False,
    )

    augmented, metadata = augmenter(video, rng=np.random.default_rng(3))
    params = metadata["augmentation_params"]["occlusion.moving_box"]
    positions = params["positions"]

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert len(positions) == video.shape[0]
    assert positions[1]["top"] >= positions[0]["top"]
    assert positions[1]["left"] >= positions[0]["left"]
    assert params["temporal_mode"] == "smooth_motion"
    assert np.any(augmented == 0)
    assert params["fill"] == "black"


def test_occlusion_config_validation():
    with pytest.raises(ValueError, match="box_area_range minimum"):
        build_augmenter("occlusion.sequence_box", box_area_range=(0.2, 0.1))

    with pytest.raises(ValueError, match="unknown border edges"):
        build_augmenter("occlusion.border_intrusion", edges=("center",))

    with pytest.raises(ValueError, match="velocity_range minimum"):
        build_augmenter("occlusion.moving_box", velocity_range=(3, -3))

    with pytest.raises(ValueError, match="fill must be 'black'"):
        build_augmenter("occlusion.sequence_box", fill="random_color")
