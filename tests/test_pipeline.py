import numpy as np

from lmfao import AugmentationPipeline, Augmenter, list_augmenters
from lmfao.base import Metadata, Video
from lmfao.registry import register_augmenter


@register_augmenter("test_passthrough")
class TestPassthrough(Augmenter):
    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        metadata.setdefault("augmentation_params", {})[self.name] = {"called": True}
        return video.copy(), metadata


def test_no_builtin_augmenters_are_registered_yet():
    assert list_augmenters() == ["test_passthrough"]


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
    assert metadata["augmentation_params"] == {"test_passthrough": {"called": True}}


def test_pipeline_is_reproducible_with_seed():
    video = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    config = [{"name": "test_passthrough", "params": {}}]

    pipeline = AugmentationPipeline.from_config(config, seed=123)
    first, first_metadata = pipeline(video)
    second, second_metadata = pipeline(video)

    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata
