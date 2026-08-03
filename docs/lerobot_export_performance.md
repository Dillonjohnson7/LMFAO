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

## Where the time actually goes (measured, warm, quiet machine, 777-frame clip)

Definitive per-category budget for one source episode (14 variants), with all
optimizations (LUT lighting, decode-once, veryfast):

| Category | # var | augment | encode | total |
|----------|-------|---------|--------|-------|
| lighting (brightness/contrast/color-temp) | 6 | 11.8 s | 14.5 s | 26.3 s |
| noise (gaussian/uniform) | 2 | **18.2 s** | 4.8 s | 23.0 s |
| occlusion (seq/border/moving box) | 4 | 0.8 s | 9.7 s | 10.5 s |
| spatial (random crop) | 2 | 0.9 s | 4.8 s | 5.8 s |
| decode (once) | — | — | — | 1.0 s |
| **TOTAL** | 14 | **31.7 s** | **33.9 s** | **66.6 s** |

Full 45-source run: **~50 min**. The split is ~48% augment / 51% encode / 1%
decode — no single dominant cause. Within augment, **noise (18.2 s) is now the
biggest** (RNG-bound; lighting dropped to 11.8 s after the LUT rewrite). Encode
(2.4 s/variant x 14) is the biggest single line and is codec-bound.

Progression: original ~2.9 h -> 72 min (decode-once + veryfast) -> ~50 min (LUT
lighting).

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
