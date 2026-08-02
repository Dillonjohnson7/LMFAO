from __future__ import annotations

from dataclasses import dataclass

from lmfao.base import KernelFeature, KernelRuntime, Metadata, Video
from lmfao.registry import register_kernel_feature


@register_kernel_feature(
    "category.feature_name",
    tags=("category", "gpu"),
    description="Short human-readable description for the central hub.",
)
@dataclass
class FeatureName(KernelFeature):
    strength: float = 1.0

    def launch(self, video: Video, runtime: KernelRuntime, metadata: Metadata) -> Metadata:
        runtime.launch_kernel(
            kernel_name="category.feature_name",
            grid=("TODO",),
            block=("TODO",),
            args=[video, self.strength],
            stream=None,
        )
        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return metadata
