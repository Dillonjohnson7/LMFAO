# Speeding up `lmfao-augment` (LeRobot dataset export)

Notes for the next full-scale run. Everything here is deferred optimization: the
first production run was left on the simple, correct path so it would finish
reliably. None of these change augmentation output semantics unless called out.

## Baseline (what we measured)

Full run of `Dillonjohnson/pick_place_v2`: 45 source episodes x 14 variants =
630 output episodes, wrist camera only (1280x720, 30fps, 472-1026 frames per
episode), on a 24 GB M-series Mac.

- **~15-17 s per output episode**, ~2.7 h total, ~8 GB output.
- Peak RSS ~11.5 GB per clip after the in-place lighting fix (`brightness`/
  `contrast` used to build 2-3 full-clip float32 copies and OOM-killed the run
  on the 1026-frame episode; see `features/lighting/*.py`).

### Where the time actually goes

Per output clip (~800 frames of 720p), roughly:

| Stage                         | Share | Why |
|-------------------------------|-------|-----|
| H.264 **re-encode** (libx264) | ~70%  | Real video compression, `preset=medium` (default), CPU. Done 630x. |
| H.264 **decode** of source    | ~25%  | And done redundantly — once *per variant*, so 14x per source episode. |
| Augmentation (numpy math)     | ~5%   | Brightness/contrast/noise on the array — cheap. |

The "filters" are not the cost. The codec is. Two structural issues drive it:
the encoder preset, and decoding each source clip 14 times.

## Optimizations, ranked by payoff

### 1. Faster libx264 preset (biggest single lever)

`encode_mp4` (`datasets/_video.py:66`) sets only CRF:

```python
stream.options = {"crf": str(crf)}       # -> libx264 preset defaults to "medium"
```

Add a preset:

```python
stream.options = {"crf": str(crf), "preset": preset}   # e.g. "veryfast"
```

- `veryfast`: ~3-4x faster encode than `medium`, ~15-25% larger files at the
  same CRF, visually near-identical. Good default for training data.
- `ultrafast`: ~6-8x faster, ~2-3x larger files. Use if disk is not the concern.
- Tradeoff is **file size / compression efficiency, not correctness** — the
  decoded frames a training loop sees are the same augmented pixels within CRF
  tolerance. Expose `preset` as a CLI flag (`--encode-preset`, default
  `veryfast`) so it is auditable.

Expected: since encode is ~70% of the time, `veryfast` alone takes the run from
~2.7 h to roughly **~50-70 min**.

### 2. Decode each source clip once, reuse across its 14 variants

`cli.run` decodes inside the variant loop (`cli.py:112`):

```python
for var_i, variant in enumerate(variants):
    def _augment(cam, cam_i, ...):
        frames = reader.read_video(ep_i, cam)   # <-- re-decodes the SAME clip every variant
```

Decode once per (episode, camera), then run all variants against the in-memory
uint8 clip:

```python
for cam in cams:
    src = reader.read_video(ep_i, cam)          # decode ONCE (~2.8 GB uint8 for 1026 frames)
    for var_i, variant in enumerate(variants):
        out = AugmentationPipeline.from_config(variant["pipeline"], seed=...)(src)[0]
        # encode out ...
```

- Eliminates 13 of every 14 decodes -> removes ~20-25% of total wall time.
- **Memory:** holds one uint8 source (~2.8 GB) plus one augmented float buffer
  (~11 GB peak) = ~14 GB. Fits in 24 GB. The current per-variant decode exists
  only to keep the lazy per-camera provider's peak minimal; with the in-place
  lighting fix the reuse form is well within budget.
- Output identical (same seeds, same source pixels).

**#1 + #2 together: ~2.7 h -> roughly ~30-45 min, no quality change worth
worrying about.**

### 3. Hardware encoder (aggressive, macOS)

Swap libx264 for Apple VideoToolbox:

```python
container.add_stream("h264_videotoolbox", rate=...)   # ASIC/GPU encode
```

- Near real-time or faster encode, low CPU. Can be 5-10x on the encode stage.
- Caveats: quality is bitrate-controlled, not CRF — set a target bitrate and
  spot-check quality. macOS only; keep libx264 as the portable fallback. Make it
  opt-in (`--encoder videotoolbox`).

### 4. Parallel episodes (bounded by RAM here)

The run is single-process. Episodes are independent, so N worker processes give
~Nx — but each worker peaks ~11-14 GB, so on 24 GB only ~1 extra worker is safe
alongside the OS and apps. Parallelism pays off on a bigger-RAM box or once
per-clip peak is lower (e.g. after chunked encoding). libx264 already uses
frame-level threads within a single encode, so CPU cores are not idle.

Worktree isolation is not needed (each worker writes distinct files), but the
`LeRobotWriter` accumulates global stats and must be sharded-then-merged if you
parallelize across the writer: have each worker emit a partial dataset, then a
final merge pass concatenates data/videos and re-aggregates `meta/stats.json`
and the episodes parquet.

### 5. Lower per-clip peak -> unlocks more parallelism (structural)

Augmenters operate on the whole clip in float32 (~11 GB for 1026 frames). A
streaming/chunked apply (process temporal blocks, thread the RNG) would bound
memory to a chunk regardless of episode length and let more workers run in
parallel. Caveat: clip-global augmenters (`occlusion.moving_box` trajectory,
`occlusion.sequence_box` constant box, `border_intrusion` constant mode) must
compute their per-clip parameters up front and then apply per chunk, or their
temporal coherence breaks. This is the real refactor; do it only if you need to
scale to many-camera / very-long-episode datasets.

## Recommended next-run config

Start with the two cheap, safe wins:

1. `preset=veryfast` in `encode_mp4` (add `--encode-preset`, default it).
2. Decode-once-per-source in `cli.run`.

That should land a full 45x14 run in **~30-45 min** with output essentially
indistinguishable from today's. Reach for VideoToolbox (#3) or parallelism
(#4/#5) only if you push to much larger datasets or multi-camera exports.

## Also worth adding (reliability, not speed)

- **Resume / checkpointing.** The CLI writes `meta/` only in `close()`, from
  in-memory accumulators. An OOM `SIGKILL` (which bypasses the `finally`) leaves
  completed episodes on disk but unfinalized and unloadable, forcing a full
  rerun. Either checkpoint the writer state every N episodes, or add `--resume`
  that rebuilds the accumulators (stats, episode rows, task map) from the
  partial output on disk and continues. This is what turned the v1 OOM into a
  from-scratch rerun.
