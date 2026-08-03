# `lmfao-augment` performance

Measured on the real `Dillonjohnson/pick_place_v2` (45 episodes x 14 variants =
630 outputs, wrist camera, 1280x720 @ 30fps, 472-1026 frames/episode) on a 24 GB
M-series Mac.

## What was built (committed)

| Change | Effect |
|--------|--------|
| **decode source once**, reuse across all 14 variants (was 14x re-decode) | removes ~13 redundant decodes/source |
| **`veryfast` encoder preset** (default), configurable via `--encode-preset` | ~2.4x faster encode than libx264's `medium` default, *and smaller files* |
| **in-place lighting math** (`brightness`/`contrast`) | peak RSS ~25 GB -> ~11.5 GB; stopped the OOM on 1026-frame clips |
| **`--resume`** (checkpoint after each source episode, atomic write) | a SIGKILL/OOM no longer forces a from-scratch rerun |
| `--encoder` / `--encode-bitrate` | opt into a hardware encoder (e.g. `h264_videotoolbox`) |

**Result: ~195 s/source -> ~96 s/source. Full 45x14 run ~2.9 h -> ~72 min, same
output size, and now crash-recoverable.** Output verified bit-for-bit identical
(trajectory, frame counts, schema).

## Where the time actually goes (measured, warm, per 777-frame clip)

An earlier guess blamed the encoder; the benchmark disproved it. The real cost is
the **augmentation**, and within it, **noise**:

| Augmenter | Time | Note |
|-----------|------|------|
| `noise.gaussian` / `noise.uniform` | **~12.5 s each** | generates 715M random floats on CPU (numpy) |
| `lighting.brightness/contrast/color_temperature` | ~3.5 s each | whole-clip float32 |
| `spatial.random_crop` | ~0.5 s | uint8 |
| `occlusion.*` | ~0.2 s | uint8 |
| decode (once) | ~4.5 s | |
| encode (veryfast) | ~2.5 s | per variant |

So per source episode now: ~48 s augment (half of it the 2 noise variants) +
~35 s encode (14x) + ~4.5 s decode. **Augment is the floor for a serial run.**

## Remaining levers (not built — each has a real tradeoff)

### A. GPU noise — DO NOT use torch/MPS here
Noise (~25 s/source) is the biggest single cost and the accelerator path
(`features/noise/accelerator.py`) already supports MPS via torch. **But
installing torch deadlocks `import av` (PyAV) on this environment** — an
OpenMP/dylib conflict that hangs the whole video pipeline. Verified: with torch
present `import av` never returns; uninstalling torch restores it. If GPU noise
is ever wanted, it must run in a **separate process/venv** that never imports
PyAV, or use a CUDA box where the conflict does not occur. Not worth it on this
Mac.

### B. Overlap encode with augment (safe, ~1.6x more -> ~45 min)
Augment is serial (numpy, 11 GB float, one clip at a time) and encode is serial
after it. Handing each variant's encode to a small thread pool (libx264 releases
the GIL; encode holds only the ~2.8 GB uint8 result, not the 11 GB float) lets
the next variant's augment overlap the previous encode. Wall ~= max(augment,
encode) ~= ~50 s/source. Memory-safe with 2-3 encode threads (~11 + 2x2.8 =
~17 GB < 24 GB). Requires decoupling encode from `add_episode`'s synchronous path.

### C. Process-parallel episodes (fast, but reintroduces OOM risk on 24 GB)
Source episodes are independent; N worker processes each writing a shard, then a
merge pass (concatenate data/videos with renumbered indices, re-aggregate
`meta/stats.json`, episodes, tasks). Near-Nx. **The catch:** each worker peaks
~11-14 GB during a float augment, so 2 concurrent big episodes (~28 GB) exceed
24 GB and OOM — exactly the failure we just removed. Only safe with a
memory-aware worker cap (effectively 1 on the big episodes here) or on a
bigger-RAM / CUDA box, where it would reach ~15-25 min. `--resume` makes an
occasional OOM recoverable, but this trades the "no failure" guarantee for speed.

## Recommendation
Serial fast path (shipped): **~72 min, memory-safe, resumable** — honors both
"much faster" and "no failure". If ~45 min is wanted without touching memory
safety, build **B**. Reserve **C** for a machine with more RAM (or a GPU), where
it is both fast and safe.
