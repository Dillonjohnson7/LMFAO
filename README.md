# LMFAO

**Lightweight Modular Feature-Based Augmentation Operation.**

A command-line tool for augmenting robot-learning datasets. Point `lmfao` at a
[LeRobot](https://github.com/huggingface/lerobot) v3 dataset — a local folder or a
Hugging Face link — and it turns each recorded episode into many varied training
clips (relit, noised, occluded, re-cropped) and writes them back out as a
ready-to-train LeRobot dataset.

Collecting robot demonstrations is the expensive part. Teleoperating an arm
through hundreds of pick-and-place episodes is slow, and a policy trained on those
clips tends to latch onto the exact lighting, background, and camera framing it
happened to see. Change the room's lighting, let the afternoon sun move, or nudge
the camera a few centimeters, and the policy that looked great in the lab falls
apart. LMFAO gets more out of the demos you already recorded — a more robust
policy without more hours on the robot, a second camera rig, or a GPU.

## The CLI

| command | what it does |
| --- | --- |
| **`lmfao`** | interactive wizard — paste a dataset link, pick effects, run. No flags to remember. |
| **`lmfao augment`** | season real footage at native resolution into many training variants (the trainable output). Magnitude sweeps, `--variants`, and crash-safe `--resume` for large runs. |
| **`lmfao inspect`** | summarize any LeRobot dataset — episodes, camera streams, tasks, missing shards — without decoding video. |
| **`lmfao generate`** | *experimental* — synthesize novel-camera-view episodes (low-res reference renderer). |

There's also a browser demo under [`web/`](web/) for previewing augmentations and
exporting a CLI job config.

Everything runs on CPU, is seeded end to end, and records every applied
augmentation in the dataset's metadata, so augmented clips drop straight back into
training and stay auditable.

## Install

```bash
git clone https://github.com/Dillonjohnson7/LMFAO.git
cd LMFAO
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,lerobot]"
```

The core library is numpy-only. The `lerobot` extra adds `pyarrow` and `av` so the
CLI can read and write real LeRobot v3 datasets (video shards + parquet).

## Quickstart

New here? Run the wizard and paste your dataset link:

```bash
lmfao
```

It asks where your data is (a Hugging Face link like
`https://huggingface.co/datasets/owner/name`, a local folder, or a built-in demo),
what you want to do, and which effects to apply — including a **magnitude sweep**
that turns "brightness, 3 steps" into ±5% / ±10% / ±15% variants and shows the
projected video count before it runs. The flag commands below do the same thing
for scripts.

```bash
# What is in this dataset? (episodes, camera streams, tasks, missing shards)
lmfao inspect ~/data/my_dataset --episodes

# Augment real footage at native resolution — 3 seasoned copies of every episode
lmfao augment --input ~/data/my_dataset --output out/aug \
    --config pipeline.json --variants 3

# Large run? --resume checkpoints after each source episode so a crash continues
lmfao augment --input ~/data/big_dataset --output out/aug \
    --config pipeline.json --variants 14 --resume
```

`pipeline.json` is the list of augmentation steps to apply:

```json
[
  {"name": "lighting.brightness", "params": {}, "probability": 1.0},
  {"name": "lighting.color_temperature", "params": {}, "probability": 1.0},
  {"name": "noise.gaussian", "params": {"sigma": 0.03}, "probability": 1.0}
]
```

Configs are validated before any data is loaded, so a typo fails in milliseconds
with a one-line error. Augmented episodes are stamped in a
`meta/lmfao_provenance.json` sidecar, so they stay distinguishable from real
footage on read-back (`lmfao inspect` shows the split). Augmentation runs at the
footage's native resolution and streams one episode at a time, so a large run
never has to hold the whole dataset in memory.

## Augmentations

- **lighting:** `lighting.brightness`, `lighting.contrast`,
  `lighting.color_temperature`
- **noise:** `noise.gaussian`, `noise.uniform`
- **occlusion:** `occlusion.sequence_box`, `occlusion.border_intrusion`,
  `occlusion.moving_box`
- **spatial:** `spatial.random_crop`

Each is an independent plug-in. To list exactly what is registered in your
install:

```python
from lmfao import list_augmenter_info

for feature in list_augmenter_info():
    print(feature.name, feature.tags, feature.description)
```

## Using LMFAO as a library

The CLI is the primary interface, but the pipeline is a plain Python API too. A
clip is a NumPy array of shape `(frames, height, width, channels)` — the same
layout a LeRobot dataset decodes to; here we fake one.

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
pip install -e ".[dev,lerobot]"
pip install pre-commit
python3 -m pre_commit install   # gitleaks on every commit (use python -m so PATH is irrelevant)
pytest
```

If `pre-commit install` refuses because `core.hooksPath` is set (Cursor cloud
agents do this), leave that alone and either scan on demand with
`python3 -m pre_commit run --all-files` or rely on the CI Secret scan job — do
not unset the agent hooks path.
