# Adding Backend-Agnostic Features

LMFAO separates feature semantics from execution.

```text
features/  = what augmentation means
backends/  = how that augmentation runs
pipeline   = which features to apply and in what order
registry   = central hub for feature discovery
```

## Feature Contract

A feature:

- inherits from `lmfao.base.AugmentationFeature`
- lives under `src/lmfao/features/<family>/`
- registers itself with `@register_feature(...)`
- receives an opaque video handle owned by the runtime
- calls `runtime.execute(...)` with an operation name and params
- records sampled params under `metadata["augmentation_params"]`
- does not import Torch, TorchVision, Pandas, CUDA, CuPy, or Triton directly
- does not loop over frames or pixels in Python

The feature should be small enough that someone can understand the augmentation
contract without reading backend code.

## Feature Template

```python
from dataclasses import dataclass

from lmfao.base import AugmentationFeature, AugmentationRuntime, Metadata, Video
from lmfao.registry import register_feature


@register_feature(
    "occlusion.random_box",
    tags=("occlusion", "image"),
    backends=("torch_cpu", "torch_cuda", "cuda"),
    description="Masks a rectangular region in each RGB observation frame.",
)
@dataclass
class RandomBoxOcclusion(AugmentationFeature):
    area: float = 0.2
    fill_value: int = 0

    def apply(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Metadata,
    ) -> tuple[Video, Metadata]:
        params = {
            "area": self.area,
            "fill_value": self.fill_value,
        }
        output = runtime.execute(
            operation_name=self.name,
            video=video,
            params=params,
            metadata=metadata,
        )
        metadata.setdefault("augmentation_params", {})[self.name] = params
        return output, metadata
```

## Backend Contract

A backend runtime:

- exposes a stable `backend` name such as `torch_cpu`, `torch_cuda`, or `cuda`
- implements `execute(operation_name, video, params, metadata, stream=None)`
- owns dependency imports and data representation details
- dispatches operation names to concrete implementations
- returns the updated video handle

CPU backends should use vectorized Torch/TorchVision/Pandas operations. GPU
backends should keep video data on device and avoid host copies in the hot path.

## Occlusion-Only Starting Point

If occlusion is the only feature today, use this shape:

```text
src/lmfao/
├── features/
│   ├── __init__.py
│   └── occlusion/
│       ├── __init__.py
│       └── random_box.py
└── backends/
    ├── __init__.py
    ├── base.py
    ├── torch_cpu.py
    └── cuda/
        ├── __init__.py
        ├── runtime.py
        └── kernels/
            └── random_box.cu
```

`features/occlusion/random_box.py` defines the augmentation name, params, and
metadata. `backends/torch_cpu.py` or `backends/cuda/runtime.py` performs the
actual operation.

## Full Planned Layout

```text
src/lmfao/features/
├── __init__.py
├── lighting/
│   ├── __init__.py
│   ├── shadow.py
│   ├── rgb_shift.py
│   ├── brightness.py
│   └── contrast.py
├── noise/
│   ├── __init__.py
│   ├── gaussian.py
│   ├── salt_pepper.py
│   └── blur.py
├── occlusion/
│   ├── __init__.py
│   ├── random_box.py
│   ├── cutout.py
│   └── mask.py
├── temporal/
│   ├── __init__.py
│   ├── speed_up.py
│   └── slow_down.py
└── geometry/
    ├── __init__.py
    ├── resize.py
    ├── rotate.py
    └── flip.py
```

Use dotted names so the central hub stays flat but browsable:

- `lighting.shadow`
- `lighting.rgb_shift`
- `noise.gaussian`
- `occlusion.random_box`
- `temporal.speed_up`
- `geometry.flip`
