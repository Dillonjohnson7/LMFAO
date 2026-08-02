from __future__ import annotations

from typing import Any, Optional

import pytest

from lmfao import AugmentationPipeline, build_feature, list_features
from lmfao.base import AugmentationRuntime, Metadata, Video


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
                "params": dict(params),
                "metadata": dict(metadata),
                "stream": stream,
            }
        )
        return video


def test_occlusion_features_register_with_dotted_names():
    features = list_features()

    assert "occlusion.sequence_box" in features
    assert "occlusion.border_intrusion" in features
    assert "occlusion.moving_box" in features


def test_sequence_box_calls_runtime_and_records_metadata():
    video = object()
    runtime = RecordingRuntime()
    pipeline = AugmentationPipeline.from_config(
        [
            {
                "name": "occlusion.sequence_box",
                "params": {
                    "box_area_range": (0.05, 0.10),
                    "aspect_ratio_range": (1.0, 1.5),
                    "fill": "mean",
                },
            }
        ]
    )

    augmented, metadata = pipeline(video, runtime)

    expected_params = {
        "box_area_range": (0.05, 0.10),
        "aspect_ratio_range": (1.0, 1.5),
        "fill": "mean",
        "temporal_mode": "sequence_constant",
    }
    assert augmented is video
    assert runtime.calls[0]["operation_name"] == "occlusion.sequence_box"
    assert runtime.calls[0]["params"] == expected_params
    assert metadata["augmentation_params"] == {"occlusion.sequence_box": expected_params}


def test_border_intrusion_calls_runtime_and_records_metadata():
    video = object()
    runtime = RecordingRuntime()
    feature = build_feature(
        "occlusion.border_intrusion",
        edges=("bottom",),
        max_fraction=0.15,
        fill=(12, 34, 56),
        temporal_mode="constant",
    )

    augmented, metadata = feature(video, runtime)

    expected_params = {
        "edges": ("bottom",),
        "max_fraction": 0.15,
        "fill": (12, 34, 56),
        "temporal_mode": "constant",
    }
    assert augmented is video
    assert runtime.calls[0]["operation_name"] == "occlusion.border_intrusion"
    assert runtime.calls[0]["params"] == expected_params
    assert metadata["augmentation_params"] == {"occlusion.border_intrusion": expected_params}


def test_moving_box_calls_runtime_and_records_metadata():
    video = object()
    runtime = RecordingRuntime()
    feature = build_feature(
        "occlusion.moving_box",
        box_area_range=(0.03, 0.12),
        velocity_range=(-4, 6),
        fill="random_color",
        edge_bounce=False,
    )

    augmented, metadata = feature(video, runtime, metadata={"demo_id": "abc"})

    expected_params = {
        "box_area_range": (0.03, 0.12),
        "aspect_ratio_range": (0.5, 2.0),
        "velocity_range": (-4.0, 6.0),
        "fill": "random_color",
        "edge_bounce": False,
        "temporal_mode": "smooth_motion",
    }
    assert augmented is video
    assert runtime.calls[0]["metadata"] == {"demo_id": "abc"}
    assert metadata["demo_id"] == "abc"
    assert metadata["augmentation_params"] == {"occlusion.moving_box": expected_params}


def test_occlusion_config_validation():
    with pytest.raises(ValueError, match="box_area_range minimum"):
        build_feature("occlusion.sequence_box", box_area_range=(0.2, 0.1))

    with pytest.raises(ValueError, match="unknown border edges"):
        build_feature("occlusion.border_intrusion", edges=("center",))

    with pytest.raises(ValueError, match="velocity_range minimum"):
        build_feature("occlusion.moving_box", velocity_range=(3, -3))
