# Adding Augmentation Features

Each feature should be independently owned and live in its own module under
`src/lmfao/features/`.

## Contract

An augmenter:

- inherits from `lmfao.base.Augmenter`
- accepts a video tensor with shape `(frames, height, width, channels)`
- returns `(augmented_video, metadata)`
- records any sampled random parameters under `metadata["augmentation_params"]`
- uses the provided `rng` instead of creating its own random generator

## Template

```python
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import register_augmenter


@register_augmenter(
    "feature_name",
    tags=("category",),
    description="Short human-readable description for the central hub.",
)
@dataclass
class FeatureName(Augmenter):
    strength: float = 1.0

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator):
        augmented = video.copy()

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return augmented, metadata
```

Add the new class to `src/lmfao/features/__init__.py` so it registers at import
time. Add focused tests for shape, dtype, metadata, and deterministic behavior
when a seed is used.

The central registry lives in `src/lmfao/registry.py`. Feature owners should
not edit its internals for normal feature work; they should use the
`@register_augmenter(...)` decorator from their own module.

## Occlusion Features

Occlusion has several related strategies, so keep it as a feature family:

```text
src/lmfao/features/
└── occlusion/
    ├── __init__.py
    ├── sequence_box.py
    ├── border_intrusion.py
    ├── moving_box.py
    └── utils.py

tests/
└── features/
    └── occlusion/
        └── test_occlusion_features.py
```

Recommended initial occlusion augmenters:

- `occlusion.sequence_box`: samples one rectangular mask and applies it to the
  same image region across all observation frames
- `occlusion.border_intrusion`: masks a strip entering from the top, bottom,
  left, or right camera border
- `occlusion.moving_box`: samples one mask and moves it smoothly through the
  observation sequence

Import the occlusion package from `src/lmfao/features/__init__.py`, and import
each occlusion module from `src/lmfao/features/occlusion/__init__.py`, so every
augmenter registers when `lmfao` is imported.

Use dotted names so the central hub stays flat but browsable:

- `lighting.shadow`
- `lighting.rgb_shift`
- `noise.gaussian`
- `occlusion.sequence_box`
- `occlusion.border_intrusion`
- `occlusion.moving_box`

## Choosing Features

The pipeline can be built from config:

```python
pipeline = AugmentationPipeline.from_config(
    [
        {
            "name": "occlusion.sequence_box",
            "params": {"box_area_range": (0.05, 0.20), "fill": "mean"},
            "probability": 0.75,
            "enabled": True,
        },
        {
            "name": "occlusion.border_intrusion",
            "params": {"edges": ("bottom",), "max_fraction": 0.2},
        }
    ],
    seed=42,
)
```

The registry exposes the central hub:

```python
from lmfao import list_augmenter_info

for feature in list_augmenter_info():
    print(feature.name, feature.tags, feature.description)
```
