# LMFAO

Lightweight Modular Feature-Based Augmentation Operation.

LMFAO expands robot imitation-learning datasets by augmenting recorded
demonstration videos. You capture a limited set of real demonstrations (for
example a SO-101 arm doing a pick-and-place task, stored as a LeRobot dataset),
and LMFAO applies composable, reproducible augmentations to those clips so a
trained policy sees far more visual variety than you physically recorded. The
goal is more robust policies without more hours on the robot.

Each augmentation is a self-contained module that registers itself by name
(lighting, occlusion, spatial crops, sensor noise, and so on). You compose them
into a pipeline from a plain config, with a per-step probability and a seed so
every run is reproducible.

## What it does

- Reads demonstration clips as NumPy arrays of shape
  `(frames, height, width, channels)`, the same layout a LeRobot dataset decodes
  to.
- Runs a configurable chain of augmentations. Each step fires with its own
  probability, so every pass produces a different but reproducible variant of the
  clip.
- Preserves shape and dtype, so an augmented clip drops straight back into a
  training set.
- Records which augmentations ran, and with what sampled parameters, in a
  metadata dict, so every synthetic frame stays auditable.

## Install

```bash
pip install -e ".[dev]"
```

## Basic usage

A clip is a NumPy array of shape `(frames, height, width, channels)`. In practice
it comes from a recorded dataset; here we fake one.

```python
import numpy as np

from lmfao import AugmentationPipeline

video = np.full((16, 64, 64, 3), 128, dtype=np.uint8)

pipeline = AugmentationPipeline.from_config(
    [
        {"name": "lighting.brightness", "params": {"min_factor": 0.7, "max_factor": 1.3}, "probability": 0.75},
        {"name": "lighting.color_temperature", "params": {"intensity": 0.35}, "probability": 0.5},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, metadata={"source": "demo"})

print(metadata["augmentations"])        # which steps actually ran this pass
print(metadata["augmentation_params"])  # the parameters they sampled
```

## Available augmentations

List everything currently registered:

```python
from lmfao import list_augmenter_info

for feature in list_augmenter_info():
    print(feature.name, feature.tags, feature.description)
```

The lighting family (`lighting.brightness`, `lighting.contrast`,
`lighting.color_temperature`) ships today. Occlusion, spatial crops, and noise
are landing as separate feature modules.

## Adding a feature

Anyone can add a feature under `src/lmfao/features/` without touching the
pipeline. Decorate an `Augmenter` subclass so it registers by name:

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
        # Modify augmented here, sampling from rng so runs stay reproducible.
        metadata.setdefault("augmentation_params", {})[self.name] = {
            "strength": self.strength,
        }
        return augmented, metadata
```

Then import it from `src/lmfao/features/__init__.py` so it registers when the
package loads. See `docs/adding_features.md` for the full contributor contract
and checklist.

## Development

```bash
pytest
```
