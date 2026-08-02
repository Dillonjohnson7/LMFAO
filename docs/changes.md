# Changes: noise augmentation

## What landed

Two augmenters, `noise.gaussian` and `noise.uniform`, plus a self-contained
accelerator that draws the random numbers on a GPU when one is present.

```text
src/lmfao/features/noise/
    __init__.py       exports both augmenters
    gaussian.py       noise.gaussian
    uniform.py        noise.uniform
    accelerator.py    all device-specific code
```

The layout follows the occlusion package: one module per feature, shared helpers
beside them. Adding a third noise type means naming a distribution in a new
module, not writing GPU code.

## Why the accelerator lives inside the feature

Generating the random numbers, not the arithmetic that follows, is what makes
noise expensive. That makes it the part worth accelerating, and it is specific
to noise, so it sits in the noise package rather than in LMFAO core.

Nothing outside `accelerator.py` imports torch or knows a device exists. Videos
go in and come out as NumPy arrays of the same shape and dtype, so `base.py`,
`pipeline.py`, and `registry.py` are untouched and torch stays an optional
dependency.

## Measured

Augmentation alone, 720p, Apple M3 Max, best of five runs:

| batch | NumPy | accelerated | speedup |
| --- | --- | --- | --- |
| 8 frames (22 MB) | 152.8 ms | 5.4 ms | 28x |
| 32 frames (88 MB) | 621.3 ms | 19.9 ms | 31x |
| 64 frames (177 MB) | 1289.6 ms | 39.1 ms | 33x |

Those accelerated timings include uploading the clip and downloading the result
on every call. Holding the video on the device between calls measures 35-43x, so
the round trip costs about 20% of the win. That was the deciding number: it is
not worth leaking device handles through the public API to recover it.

End to end through `examples/augment_folder.py` on 11 real 720p clips, the CPU
path took 2068 s. The accelerated path processes a clip in roughly 40 s against
roughly 188 s, about 5x. The gap between 30x and 5x is decode and encode, which
now dominate: the augmentation is no longer the bottleneck.

## Decisions worth knowing

**Randomness still flows from the pipeline's rng.** The contract says features
use the provided `rng` rather than making their own. The device generator is
seeded from it, so a seeded run stays reproducible. The CPU and the device draw
different values for the same seed because they use different generators; each
is reproducible against itself, and both were checked to produce the same
distribution (Gaussian std 12.40 vs 12.37 at `sigma=0.05`, against 12.75
expected).

**Rounding before the dtype cast.** `preserve_dtype` casts, and casting
truncates toward zero, which biases every frame darker by about half a level.
The noise modules round first. This is kept local rather than changed in
`base.py`, since lighting and occlusion depend on the shared version.

**Unsupported dtypes fall back silently, bad devices fail loudly.** `float64`
cannot go to Metal, so `device="auto"` quietly uses the CPU for it. Asking for
`device="cuda"` on a machine without CUDA raises, because silently running 30x
slower is worse than an error.

**Strength is a fraction of the dynamic range.** `sigma=0.05` means the same
visible amount of noise on uint8 and float video, so a config ports between
datasets without retuning.

## Verified

- 34 tests pass; `ruff check src tests` is clean
- shape, dtype, and metadata preserved on both paths, across uint8, int16,
  float32, and float64
- input arrays are never mutated
- no wraparound when clipping at 0 and 255
- noise is resampled per frame, not held constant across a clip
- seeded pipeline runs reproduce exactly, composed with `lighting.brightness`

## Also in this change

`examples/augment_folder.py` batch-augments a folder through any registered
feature, so lighting, spatial, and occlusion work in it without edits.

The package layout matches the lighting, occlusion, and spatial packages that
landed while this was being written: a package per category, one module per
feature, shared helpers beside them. `accelerator.py` fills the role their
`utils.py` does.

## Not done

- CUDA is written but untested; no NVIDIA hardware here. The torch path covers
  both, so it should work, but it needs a real run before anyone trusts it.
- The `feature/gpu-kernel-skeleton` branch proposes a shared `backends/` layer.
  If that lands, this accelerator should move under it rather than staying in
  the feature package.
