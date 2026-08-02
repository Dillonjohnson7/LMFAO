# LMFAO

Lightweight Modular Feature-Based Augmentation Operation.

LMFAO augments robot demonstration datasets for ACT-style policy training. It
expands a dataset by applying image-only transforms to RGB observation frames
while leaving actions and action chunks unchanged.

The library is backend-agnostic: feature modules define what augmentation should
happen, while backend runtimes decide how to execute it on CPU or GPU.

## Dataset Contract

LMFAO expects episodes shaped like:

```text
observations: O0 ... On
actions:      a0 ... an
chunks:       action chunks used by ACT-style policies
```

Image augmentations touch only observation images. Robot state, actions, and
action chunks are carried through unchanged unless a future sequence-level
transform explicitly documents otherwise.

## Basic Usage

Video data is an opaque handle owned by the selected runtime. That handle can be
a Torch tensor, TorchVision video tensor, Pandas-backed batch record, CUDA
buffer, CuPy array, Triton allocation, or another project-specific object.

```python
from lmfao import AugmentationPipeline

video = runtime.load_video(...)

pipeline = AugmentationPipeline.from_config(
    [
        {"name": "occlusion.random_box", "params": {"area": 0.2}, "probability": 0.75},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, runtime, metadata={"episode_id": "demo-001"})
```

## Team Modules

Initial feature ownership:

- Dillon: `features/lighting/` for lighting and color channel transforms
- Andrew: `features/noise/` for noise injection
- Ryan: `features/occlusion/` for occlusions

Planned later modules:

- `features/temporal/` for slowing and speeding sequences
- `features/geometry/` for resolution changes, rotation, and flipping

Temporal and geometry transforms need extra care because they may affect
sequence alignment or camera geometry. Image-only transforms should preserve ACT
labels by default.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/architecture.md` and `docs/adding_features.md` before implementing a
feature or backend.
