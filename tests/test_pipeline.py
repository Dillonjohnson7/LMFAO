import numpy as np

from lmfao import AugmentationPipeline, Augmenter, list_augmenter_info, list_augmenters
from lmfao.base import Metadata, Video
from lmfao.registry import register_augmenter


@register_augmenter("test_passthrough", tags=("test",), description="A test-only passthrough augmenter.")
class TestPassthrough(Augmenter):
    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        metadata.setdefault("augmentation_params", {})[self.name] = {"called": True}
        return video.copy(), metadata


def test_registered_augmenters_are_visible_in_hub():
    assert list_augmenters() == [
        "lighting.brightness",
        "lighting.color_temperature",
        "lighting.contrast",
        "test_passthrough",
    ]

    info_by_name = {info.name: info for info in list_augmenter_info()}

    passthrough = info_by_name["test_passthrough"]
    assert passthrough.tags == ("test",)
    assert passthrough.description == "A test-only passthrough augmenter."

    for name in ("lighting.brightness", "lighting.color_temperature", "lighting.contrast"):
        assert "lighting" in info_by_name[name].tags


def test_pipeline_preserves_shape_and_dtype():
    video = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    pipeline = AugmentationPipeline.from_config(
        [
            {"name": "test_passthrough", "params": {}},
        ],
        seed=7,
    )

    augmented, metadata = pipeline(video)

    assert augmented.shape == video.shape
    assert augmented.dtype == video.dtype
    assert metadata["augmentations"] == ["test_passthrough"]
    assert metadata["skipped_augmentations"] == []
    assert metadata["augmentation_params"] == {"test_passthrough": {"called": True}}


def test_pipeline_is_reproducible_with_seed():
    video = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    config = [{"name": "test_passthrough", "params": {}}]

    pipeline = AugmentationPipeline.from_config(config, seed=123)
    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata


def test_pipeline_can_skip_feature_by_probability():
    video = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    pipeline = AugmentationPipeline.from_config(
        [{"name": "test_passthrough", "params": {}, "probability": 0.0}],
        seed=123,
    )

    augmented, metadata = pipeline(video)

    np.testing.assert_array_equal(augmented, video)
    assert metadata["augmentations"] == []
    assert metadata["skipped_augmentations"] == ["test_passthrough"]
