# LMFAO

Lightweight Modular Feature-Based Augmentation Operation.

LMFAO is a small central library for GPU-backed video data augmentation. Each
feature lives in its own module, registers itself by name, and contributes a
thin kernel launch wrapper that can be selected through a shared pipeline.

## Install

```bash
pip install -e ".[dev]"
```

## Basic Usage

Videos should stay GPU-resident. The core package treats the video as an opaque
handle owned by your runtime, such as a CUDA buffer, PyTorch tensor, CuPy array,
or custom device allocation.

Example:

```python
from lmfao import KernelPipeline

video = runtime.upload_video(...)

pipeline = KernelPipeline.from_config(
    [
        # Add registered feature configs here, for example:
        # {"name": "lighting.shadow", "params": {"strength": 0.5}, "probability": 0.75},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, runtime, metadata={"source": "demo"})
```

No feature implementations are included yet. Lighting changes, occlusions,
noise, and other augmentations should be added as separate GPU kernel features.

## Adding a Feature

Each person can add a feature in `src/lmfao/features/` without editing the
pipeline.

```python
from dataclasses import dataclass

from lmfao.base import KernelFeature, KernelRuntime, Metadata, Video
from lmfao.registry import register_kernel_feature


@register_kernel_feature(
    "lighting.my_feature",
    tags=("lighting", "gpu"),
    description="Short description shown by the central feature hub.",
)
@dataclass
class MyFeature(KernelFeature):
    strength: float = 1.0

    def launch(self, video: Video, runtime: KernelRuntime, metadata: Metadata):
        runtime.launch_kernel(
            kernel_name="lighting.my_feature",
            grid=("TODO",),
            block=("TODO",),
            args=[video, self.strength],
            stream=None,
        )
        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return metadata
```

Then import it from `src/lmfao/features/__init__.py` so it is registered when
the package loads.

See `docs/adding_features.md` for the contributor contract and checklist.

## Development

Run tests:

```bash
pytest
```
