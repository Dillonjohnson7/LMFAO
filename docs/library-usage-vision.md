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

## Why noise got a GPU path first (landed)

Noise is dominated by drawing the random numbers, not by the arithmetic that
follows. That made it the obvious first candidate for the GPU: the whole clip is
sampled as one tensor, so a single launch covers every frame, pixel, and
channel, and the distribution is the only thing that differs between noise
types. This shipped in `features/noise/accelerator.py`.

The same observation drove the CPU path, which now draws its blocks in parallel
instead of on one core. That narrowed the accelerator's lead on a 32-frame 720p
clip from ~25x to ~4x, so the GPU is a speedup rather than a requirement --
which matters, because torch cannot be installed alongside `av` in the
dataset-touching environment.

A general device-resident array type is still future work: today the accelerator
uploads and downloads per call. That belongs in the `backends/` layer described
in `architecture.md`, not in the feature modules.

## How this maps to the current codebase

The package already has the **internals**:

- `Augmenter` base class
- Feature registry (`@register_augmenter`)
- Config-driven `AugmentationPipeline`

The helpers above should sit on top of that. Most users only touch `import lmfao` and the category functions. Power users can still use `AugmentationPipeline.from_config(...)`.

## Still to build

- Top-level helpers: `lighting`, `noise`, `occlusion`, `full_augmentations`
- Named presets (e.g. `"shadow"`, `"gaussian"`, `"cutout"`)
- Real feature implementations (noise, lighting, occlusion and spatial have all landed)
- A device-resident array type, so a multi-step pipeline is not copied per call
- Speed path for `full_augmentations` (ideally fused, not three separate copies)
