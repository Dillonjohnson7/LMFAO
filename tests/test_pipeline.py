from dataclasses import dataclass
from typing import Any, Optional

from lmfao import AugmentationFeature, AugmentationPipeline, list_feature_info, list_features
from lmfao.base import AugmentationRuntime, Metadata, Video
from lmfao.registry import register_feature


class RecordingRuntime(AugmentationRuntime):
    backend = "torch_cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        operation_name: str,
        video: Video,
        params: dict[str, Any],
        metadata: Metadata,
        stream: Optional[Any] = None,
    ) -> Video:
        self.calls.append(
            {
                "operation_name": operation_name,
                "video": video,
                "params": params,
                "metadata": dict(metadata),
                "stream": stream,
            }
        )
        return video


@register_feature(
    "test.passthrough",
    tags=("test",),
    backends=("torch_cpu", "cuda"),
    description="A test-only augmentation feature.",
)
@dataclass
class DummyPassthrough(AugmentationFeature):
    value: int = 1

    def apply(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Metadata,
    ) -> tuple[Video, Metadata]:
        params = {"value": self.value}
        output = runtime.execute(
            operation_name=self.name,
            video=video,
            params=params,
            metadata=metadata,
        )
        metadata.setdefault("augmentation_params", {})[self.name] = params
        return output, metadata


def test_builtin_occlusion_features_are_registered():
    features = list_features()
    assert "occlusion.border_intrusion" in features
    assert "occlusion.moving_box" in features
    assert "occlusion.sequence_box" in features
    assert "test.passthrough" in features
    info = next(item for item in list_feature_info() if item.name == "test.passthrough")
    assert info.name == "test.passthrough"
    assert info.tags == ("test",)
    assert info.backends == ("torch_cpu", "cuda")
    assert info.description == "A test-only augmentation feature."


def test_pipeline_executes_configured_feature_on_runtime():
    video = object()
    runtime = RecordingRuntime()
    pipeline = AugmentationPipeline.from_config(
        [
            {"name": "test.passthrough", "params": {"value": 7}},
        ],
        seed=7,
    )

    augmented, metadata = pipeline(video, runtime)

    assert augmented is video
    assert runtime.calls == [
        {
            "operation_name": "test.passthrough",
            "video": video,
            "params": {"value": 7},
            "metadata": {},
            "stream": None,
        }
    ]
    assert metadata["augmentations"] == ["test.passthrough"]
    assert metadata["skipped_augmentations"] == []
    assert metadata["augmentation_params"] == {"test.passthrough": {"value": 7}}


def test_pipeline_is_reproducible_with_seed():
    video = object()
    config = [{"name": "test.passthrough", "params": {}}]

    pipeline = AugmentationPipeline.from_config(config, seed=123)
    first_runtime = RecordingRuntime()
    second_runtime = RecordingRuntime()
    first, first_metadata = pipeline(video, first_runtime)
    second, second_metadata = pipeline(video, second_runtime)

    assert first is second
    assert first_metadata == second_metadata
    assert first_runtime.calls == second_runtime.calls


def test_pipeline_can_skip_feature_by_probability():
    video = object()
    runtime = RecordingRuntime()
    pipeline = AugmentationPipeline.from_config(
        [{"name": "test.passthrough", "params": {}, "probability": 0.0}],
        seed=123,
    )

    augmented, metadata = pipeline(video, runtime)

    assert augmented is video
    assert runtime.calls == []
    assert metadata["augmentations"] == []
    assert metadata["skipped_augmentations"] == ["test.passthrough"]


def test_feature_rejects_unsupported_backend():
    class PandasRuntime(RecordingRuntime):
        backend = "pandas_cpu"

    video = object()
    pipeline = AugmentationPipeline.from_config([{"name": "test.passthrough", "params": {}}])

    try:
        pipeline(video, PandasRuntime())
    except ValueError as exc:
        assert "does not support backend 'pandas_cpu'" in str(exc)
    else:
        raise AssertionError("expected unsupported backend error")
