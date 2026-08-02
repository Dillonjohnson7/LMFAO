from dataclasses import dataclass
from typing import Any, Optional

from lmfao import KernelFeature, KernelPipeline, list_kernel_feature_info, list_kernel_features
from lmfao.base import KernelRuntime, Metadata, Video
from lmfao.registry import register_kernel_feature


class RecordingRuntime(KernelRuntime):
    def __init__(self) -> None:
        self.launches: list[dict[str, Any]] = []

    def launch_kernel(
        self,
        kernel_name: str,
        grid: Any,
        block: Any,
        args: list[Any],
        stream: Optional[Any] = None,
    ) -> None:
        self.launches.append(
            {
                "kernel_name": kernel_name,
                "grid": grid,
                "block": block,
                "args": args,
                "stream": stream,
            }
        )


@register_kernel_feature("test.passthrough", tags=("test",), description="A test-only kernel feature.")
@dataclass
class DummyPassthrough(KernelFeature):
    value: int = 1

    def launch(self, video: Video, runtime: KernelRuntime, metadata: Metadata) -> Metadata:
        runtime.launch_kernel(
            kernel_name="test.passthrough",
            grid=(1,),
            block=(1,),
            args=[video, self.value],
            stream=None,
        )
        metadata.setdefault("augmentation_params", {})[self.name] = {"value": self.value}
        return metadata


def test_no_builtin_kernel_features_are_registered_yet():
    assert list_kernel_features() == ["test.passthrough"]
    [info] = list_kernel_feature_info()
    assert info.name == "test.passthrough"
    assert info.tags == ("test",)
    assert info.description == "A test-only kernel feature."


def test_pipeline_launches_configured_kernel_feature():
    video = object()
    runtime = RecordingRuntime()
    pipeline = KernelPipeline.from_config(
        [
            {"name": "test.passthrough", "params": {"value": 7}},
        ],
        seed=7,
    )

    augmented, metadata = pipeline(video, runtime)

    assert augmented is video
    assert runtime.launches == [
        {
            "kernel_name": "test.passthrough",
            "grid": (1,),
            "block": (1,),
            "args": [video, 7],
            "stream": None,
        }
    ]
    assert metadata["augmentations"] == ["test.passthrough"]
    assert metadata["skipped_augmentations"] == []
    assert metadata["augmentation_params"] == {"test.passthrough": {"value": 7}}


def test_pipeline_is_reproducible_with_seed():
    video = object()
    config = [{"name": "test.passthrough", "params": {}}]

    pipeline = KernelPipeline.from_config(config, seed=123)
    first_runtime = RecordingRuntime()
    second_runtime = RecordingRuntime()
    first, first_metadata = pipeline(video, first_runtime)
    second, second_metadata = pipeline(video, second_runtime)

    assert first is second
    assert first_metadata == second_metadata
    assert first_runtime.launches == second_runtime.launches


def test_pipeline_can_skip_feature_by_probability():
    video = object()
    runtime = RecordingRuntime()
    pipeline = KernelPipeline.from_config(
        [{"name": "test.passthrough", "params": {}, "probability": 0.0}],
        seed=123,
    )

    augmented, metadata = pipeline(video, runtime)

    assert augmented is video
    assert runtime.launches == []
    assert metadata["augmentations"] == []
    assert metadata["skipped_augmentations"] == ["test.passthrough"]
