# LMFAO

Lightweight Modular Feature-Based Augmentation Operation.

Collecting robot demonstrations is the expensive part. Teleoperating an arm
through hundreds of pick-and-place episodes is slow and tedious, and a policy
trained on those clips tends to latch onto the exact lighting, background, and
camera framing it happened to see. Change the room's lighting, let the afternoon
sun move, or nudge the camera a few centimeters, and the policy that looked
great in the lab falls apart.

LMFAO gets more out of the demonstrations you already recorded. It turns each
clip into many varied training examples (relit, partially occluded, re-cropped,
and so on), so your policy learns the task instead of memorizing the scene. You
get a more robust policy without more hours on the robot, a second camera rig, or
a GPU: the augmentations are cheap, run on CPU, and work on the data you already
have.

## Why it helps

- **More data from the same demos.** One recorded episode becomes many training
  clips, so you spend less time teleoperating and more time training.
- **Policies that survive the real world.** Randomized lighting, occlusion, and
  framing stop a policy from overfitting to the one scene it was recorded in.
- **Cheap and reproducible.** Runs on CPU, seeded end to end. Shape and
  dtype are preserved and every applied augmentation is recorded in metadata, so
  augmented clips drop straight back into training and stay auditable.
- **Modular by design.** Every effect is an independent plug-in. Add or swap one
  without touching the pipeline or anyone else's feature.

## Install

```bash
pip install -e ".[dev]"
```

## Basic usage

A clip is a NumPy array of shape `(frames, height, width, channels)`, the same
layout a LeRobot dataset decodes to. In practice it comes from a recorded
dataset; here we fake one.

```python
import numpy as np

from lmfao import AugmentationPipeline

video = np.full((16, 64, 64, 3), 128, dtype=np.uint8)

pipeline = AugmentationPipeline.from_config(
    [
        {"name": "lighting.brightness", "params": {"min_factor": 0.7, "max_factor": 1.3}, "probability": 0.75},
        {"name": "lighting.color_temperature", "params": {"intensity": 0.35}, "probability": 0.5},
        {"name": "occlusion.moving_box", "probability": 0.5},
        {"name": "spatial.random_crop", "probability": 1.0},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, metadata={"source": "demo"})

print(metadata["augmentations"])        # which steps actually ran this pass
print(metadata["augmentation_params"])  # the parameters they sampled
```

Change the `seed` and each pass gives you a fresh variant of the same clip; keep
it fixed and the run is byte-for-byte repeatable.

## Available augmentations

Shipping today:

- **lighting:** `lighting.brightness`, `lighting.contrast`,
  `lighting.color_temperature`
- **noise:** `noise.gaussian`, `noise.uniform`
- **occlusion:** `occlusion.sequence_box`, `occlusion.border_intrusion`,
  `occlusion.moving_box`
- **spatial:** `spatial.random_crop`

Noise draws its random numbers on a GPU when one happens to be available, which
is roughly 30x faster than NumPy. Nothing changes at the call site and no GPU is
required.

Further effects are on the way. To list exactly what is registered in your
install:

```python
from lmfao import list_augmenter_info

for feature in list_augmenter_info():
    print(feature.name, feature.tags, feature.description)
```

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
