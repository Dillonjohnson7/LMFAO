# LMFAO

Lightweight Modular Feature-Based Augmentation Operation.

LMFAO is a small central library for video data augmentation. Each augmentation
feature lives in its own module, registers itself by name, and can be composed
with other features through a shared pipeline.

## Install

```bash
pip install -e ".[dev]"
```

## Basic Usage

Videos are represented as NumPy arrays with shape:

```text
(frames, height, width, channels)
```

Example:

```python
import numpy as np

from lmfao import AugmentationPipeline

video = np.full((16, 64, 64, 3), 128, dtype=np.uint8)

pipeline = AugmentationPipeline.from_config(
    [
        # Add registered feature configs here, for example:
        # {"name": "lighting", "params": {"strength": 0.5}, "probability": 0.75},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, metadata={"source": "demo"})
```

No feature implementations are included yet. Lighting changes, occlusions,
noise, and other augmentations should be added as separate feature modules.

## Adding a Feature

Each person can add a feature in `src/lmfao/features/` without editing the
pipeline.

```python
from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import register_augmenter


@register_augmenter(
    "my_feature",
    tags=("lighting",),
    description="Short description shown by the central feature hub.",
)
@dataclass
class MyFeature(Augmenter):
    strength: float = 1.0

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator):
        augmented = video.copy()
        # Modify augmented here.
        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return augmented, metadata
```

Then import it from `src/lmfao/features/__init__.py` so it is registered when
the package loads.

See `docs/adding_features.md` for the contributor contract and checklist.

## Development

Run tests:

```bash
pytest
```
