# Adding GPU Kernel Features

Each feature should be independently owned and live in its own module under
`src/lmfao/features/`.

## Contract

A kernel feature:

- inherits from `lmfao.base.KernelFeature`
- receives a GPU-resident video handle, not a NumPy array
- launches one or more kernels through the provided `KernelRuntime`
- does not copy frames to the CPU in the hot path
- avoids Python per-frame or per-pixel loops
- records launch parameters under `metadata["augmentation_params"]`
- keeps allocations, synchronization, and stream ownership explicit

## Template

```python
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
```

Add the new class to `src/lmfao/features/__init__.py` so it registers at import
time. Add focused tests that verify the expected kernel name, launch args,
metadata, and deterministic feature selection when a seed is used.

The central registry lives in `src/lmfao/registry.py`. Feature owners should
not edit its internals for normal feature work; they should use the
`@register_kernel_feature(...)` decorator from their own module.

Use `docs/feature_template.py` as a copy/paste starting point.

## Recommended Layout

```text
src/lmfao/features/
├── __init__.py
├── lighting/
│   ├── __init__.py
│   ├── shadow.py
│   ├── shadow.cu
│   ├── rgb_shift.py
│   └── rgb_shift.cu
├── occlusion/
│   ├── __init__.py
│   ├── random_box.py
│   └── random_box.cu
└── noise/
    ├── __init__.py
    ├── gaussian.py
    └── gaussian.cu
```

Use dotted feature names so the central hub remains flat but browsable:

- `lighting.shadow`
- `lighting.rgb_shift`
- `occlusion.random_box`
- `noise.gaussian`

## Choosing Features

The pipeline can be built from config:

```python
pipeline = KernelPipeline.from_config(
    [
        {
            "name": "lighting.shadow",
            "params": {"strength": 0.5},
            "probability": 0.75,
            "enabled": True,
        }
    ],
    seed=42,
)
```

The registry exposes the central hub:

```python
from lmfao import list_kernel_feature_info

for feature in list_kernel_feature_info():
    print(feature.name, feature.tags, feature.description)
```

## Lightweight Rules

- Python should only choose features and launch kernels.
- Keep video data on the GPU.
- Reuse buffers and streams in the runtime.
- Prefer fused kernels for multiple simple pixelwise operations when launch
  overhead or memory bandwidth becomes the bottleneck.
- Only run kernels concurrently when they write to independent buffers or
  disjoint memory regions, or when the runtime explicitly resolves ordering.
