# Library Usage Vision

LMFAO is a Python library for fast video data augmentation — lighting, noise, and occlusion. Simple to import and call, built to be fast under the hood.

## Goal

One-liners after `pip install`:

```python
import lmfao

out = lmfao.lighting(data, "shadow")
out = lmfao.noise(data, "gaussian")
out = lmfao.occlusion(data, "cutout")

out = lmfao.full_augmentations(data)  # curated default stack
```

- Individual helpers: pick a category + named preset
- `full_augmentations`: professional default recipe (order, strengths, probabilities)
- Optional knobs (`seed=`, strength overrides, `return_metadata=True`) when you need them

This is just the library’s public surface — functions you call in your training or preprocessing code. Not a web/service API.

## Data format

Videos are arrays shaped:

```text
(frames, height, width, channels)
```

NumPy arrays today. A GPU array type is the obvious next step.

## Why GPU is on the roadmap

Noise is dominated by drawing the random numbers, not by the arithmetic that
follows. That makes it a good fit for the GPU: the whole clip can be sampled as
one tensor, so a single cuRAND launch covers every frame, pixel, and channel.
The distribution is the only thing that differs between noise types, so each new
type stays a single generator call.

Whenever that lands, it should run wherever the video already lives, so nothing
is copied between devices behind your back. That belongs in the `backends/`
layer described in `architecture.md`, not in the feature modules.

## How this maps to the current codebase

The package already has the **internals**:

- `Augmenter` base class
- Feature registry (`@register_augmenter`)
- Config-driven `AugmentationPipeline`

The helpers above should sit on top of that. Most users only touch `import lmfao` and the category functions. Power users can still use `AugmentationPipeline.from_config(...)`.

## Still to build

- Top-level helpers: `lighting`, `noise`, `occlusion`, `full_augmentations`
- Named presets (e.g. `"shadow"`, `"gaussian"`, `"cutout"`)
- Real feature implementations (noise has landed; lighting and occlusion are in flight)
- A GPU backend so sampling is not CPU-bound
- Speed path for `full_augmentations` (ideally fused, not three separate copies)
