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

## Choosing Features

The pipeline can be built from config:

```python
pipeline = AugmentationPipeline.from_config(
    [
        {
            "name": "feature_name",
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
from lmfao import list_augmenter_info

for feature in list_augmenter_info():
    print(feature.name, feature.tags, feature.description)
```
