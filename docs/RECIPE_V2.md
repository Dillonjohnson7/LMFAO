# Recipe v2: lifting nominal (stock-condition) success

Date: 2026-08-09. Supersedes the v1 bucket line for the nominal-performance
goal; v1 buckets remain valid for the held-out robustness line.

## Run summary

| | **stock** (control) | **v2_steady** | **v2_jitter_occlusion** |
|---|---|---|---|
| Role | control, same as v1 stock | nominal-performance arm | robustness arm |
| Brightness / contrast / color temp | — | ✓ | — |
| Gaussian noise (σ=0.03) | — | ✓ | — |
| Shot / compression / blur noise (p=0.5 / 0.5 / 0.3) | — | ✓ | — |
| Steady camera shift (±4 px, one fixed offset per episode) | — | ✓ | — |
| Per-frame camera shake (±8 px jitter) | — | — | ✓ |
| Border occluder (p=0.7) | — | — | ✓ |
| Fixed black box (p=0.5) | — | — | ✓ |
| Moving black box (p=0.5) | — | — | ✓ |
| Data per source demo | raw dataset as-is | 2 originals + 1 augmented | 2 originals + 1 augmented |
| Stock/augmented mix | 100% real | 67% / 33% | 67% / 33% |
| Episodes (from 40 demos) | 40 | 120 | 120 |
| Frames (~650/ep) | ~26k | ~78k | ~78k |
| Passes per real-demo frame @100k steps | ~31× | ~21× | ~21× |
| Passes per augmented frame @100k steps | — | ~10× | ~10× |
| Policy / steps / batch | ACT ~52M / 100k / 8 | same | same |
| lr / seed / chunk | 1e-5 / 7 / 100 | same | same |
| Wall-clock per run | ~4.5 h @ ~6.4 steps/s | same | same |
| Physical eval | 15–20 nominal trials | same | same |

All gates and the scoring harness are identical to the v1 runs.

## Why v1 could not lift the nominal number

Frame-by-frame audit of all 14 scored failure clips (§11.0 of
`policytraining_v3.md`) plus joint-pose analysis of every demo and rollout
episode (`so101/eval/diagnose_close.py` on the NUC) established:

1. **Every verified failure is the same ~1 cm missed close**, followed by an
   empty-jaw place script. No policy fails anywhere else in the task.
2. **The close is dead-reckoned**: ACT commits a 100-action chunk (3.3 s)
   from one observation, so the sub-second close is executed open-loop.
3. **v1 augmentation is invariance training**: pixel-only transforms with
   unchanged action labels teach "same action despite pixel changes." The
   bottleneck demands the opposite — sensitivity to sub-cm puck displacement.
   Two v1 ingredients plausibly hurt precision:
   - `spatial.random_crop` (per-frame ±8 px jitter ≈ ±7–8 mm on the front
     camera — exactly the miss scale — paired with unchanged labels), and
   - occlusion boxes that can cover the wrist-view close, teaching blind
     closing. (Occlusion's failures share a systematic close-pose offset,
     t≈5; stock/full fail randomly. Small n — suggestive, not settled.)

## Design rules for the steady arm

- **Position-preserving transforms only**: lighting and noise change
  appearance without moving a single pixel of task geometry, so the
  vision→action labels stay exactly consistent.
- **Geometry, if present, must be episode-constant and small**: a stable
  ±4 px offset simulates a camera remount between sessions (a real failure
  mode worth robustness) without per-frame label noise. That is what the new
  `temporal_mode="constant"` on `spatial.random_crop` provides
  (`src/lmfao/features/spatial/random_crop.py`); `per_frame` remains the
  default so v1 configs are unchanged.
- **No occluders** — they train closing without looking.

The jitter+occlusion arm deliberately breaks all three rules; it exists to
train robustness and to put a number on what that robustness costs nominal
precision.

## New buckets

Two total:

- `configs/training/buckets/v2_steady.json` — brightness + contrast + color
  temperature + gaussian σ=0.03 (always), plus shot / compression / blur
  noise (p=0.5/0.5/0.3, so each variant gets a different mild mix), plus an
  episode-constant ±4 px crop (remount simulation). Everything in this
  bucket is position-preserving; the primary nominal-performance recipe.
- `configs/training/buckets/v2_jitter_occlusion.json` — the aggressive arm:
  v1-style per-frame ±8 px crop jitter plus the three occluders (border
  intrusion p=0.7, sequence box p=0.5, moving box p=0.5). Expected to cost
  some nominal precision (see "Why v1 could not lift the nominal number");
  kept because it is the strongest robustness training of the two — the
  nominal-vs-held-out gap between the two buckets is itself a measurement of
  the precision/robustness trade-off.

Build with the existing script (defaults now produce the v2 ratio):

```bash
STOCK=/path/to/new_demos BUCKETS="v2_steady v2_jitter_occlusion" \
  ./scripts/augment_eval_buckets.sh
```

## Data ratio: 67% stock / 33% augmented

v1 buckets were ~11% original / ~89% augmented (1 original + 8 variants per
source) — the policy trained nine-tenths on seasoned data for a test that is
run on the unseasoned distribution, and each unique frame was drawn only
~3.4× in 100k steps (vs ~31× for stock's 40-episode dataset), so the buckets
converged shallowly (final losses 0.061–0.068 vs stock's 0.053).

v2 flips the anchor: **2 original copies + 1 variant per source = 67% stock /
33% augmented**, via the new `--original-copies` flag (`lmfao augment
--include-original --original-copies 2 --variants 1`). Per unique frame at
the same 100k-step budget, that is ~14 passes over each unique original and
~6.8 passes over each unique variant — double v1's augmented repetition. The
4-copies/2-variants form was rejected: same ratio and same nominal
repetition, but it halves augmented repetition (back to 3.4×) and doubles
disk for no gain. Copies are stamped `original_copy` in episode metadata so
provenance stays distinguishable.

## Recollection spec (the data half of the recipe)

Augmentation cannot add precision; only data can. Record ~40 demos of the
same task, same cameras, same start-pose procedure (`goto_start.py`,
`warmup_s=4`):

1. **Recovery demos (~8 of 40)** — deliberately miss the first close
   (offset ~1 cm), then re-approach and complete the pick. Every scored
   failure on record is a missed close with no re-attempt; ep9 proved a
   second approach converts. This is the single highest-value change.
2. **Dense near nominal (~24 of 40)** — puck within ~±5 cm of the eval
   placement; v3's 40 demos spanned the whole workspace (close poses ranged
   63–75° in shoulder_lift/elbow), leaving sparse local density at the eval
   spot. Remaining ~8 keep the wider spread for robustness.
3. **Slow the close** — dwell ~1 s with open jaws over the puck before
   closing. The close is a tiny fraction of each episode's frames; more
   close-window frames = more training signal exactly where the policy fails.

## Training plan

- 3 policies: `stock` (control), `v2_steady`, `v2_jitter_occlusion`.
- VARIANTS=1 + ORIGINAL_COPIES=2 → 40 × (2+1) = 120 episodes per augmented
  bucket (~78k frames at v3's ~650 frames/episode — a third of the
  234k-frame envelope that trained without OOM in the v1 run).
- Unchanged: ACT, 100k steps, batch 8, seed 7, lr 1e-5.
- Do **not** raise chunk_size (a 120-step chunk was floated in CHECKPOINT.md;
  longer open-loop commitment is the wrong direction for a precision failure).

## Inference lever (test before recollection — it is free)

`n_action_steps` is an inference-time knob: re-plan every 25 ticks instead of
committing all 100. `so101/eval/offline_eval.py` already supports it
(`NSTEP=25`), and `creep_test.py` runs NSTEP=25 physically.

1. Offline: `NSTEP=25` vs `NSTEP=100` on the existing stock policy — zero
   robot time.
2. If offline MAE improves, physical A/B: 15 trials each on the existing
   stock policy before spending recollection effort.
3. Keep every policy in a comparison on the **same** NSTEP — that preserves
   the `load_policy.sh` parity contract (the guard compares policy configs;
   the rollout override applies equally to all arms).

## Success criteria

- Primary: nominal success rate, v2 buckets vs v2 stock control, 15–20
  trials per policy.
- Secondary: close-pose scatter at the grasp (`diagnose_close.py`) — the v2
  policies should show tighter scatter than v1's 1.5–6° per joint.
- The held-out v1 line is untouched: once nominal is lifted, rerun the
  §11.2 conditions to measure robustness on top of the higher base rate.

## Day-of runbook (record → trained policies)

All NUC commands assume `cd ~/LMFAO/repo` and the repo pulled to at least
the commit adding `--original-copies` and `temporal_mode="constant"`.

```bash
# 1. RECORD (NUC) — NEW dataset root; do not overwrite pick_place_v3.
#    Spec: 8 recovery (miss ~1 cm, re-approach, finish) · 24 puck near the
#    eval spot · 8 wider spread. ~1 s open-jaw dwell over the puck before
#    closing. Same start-pose routine (goto_start) as the eval harness.
DATASET_ROOT=~/LMFAO/so101/datasets/pick_place_v4 ./so101/rec 40

# 2. BUILD BUCKETS (NUC) — stock symlink + the two v2 buckets,
#    2 originals + 1 variant per source (67/33) by default.
STOCK=~/LMFAO/so101/datasets/pick_place_v4 \
OUT_ROOT=~/LMFAO/so101/datasets/eval_buckets_v4 \
BUCKETS="v2_steady v2_jitter_occlusion" \
./scripts/augment_eval_buckets.sh

# 3. SANITY-CHECK THE MIX (NUC) — each bucket: 120 episodes, ~2/3 stamped
#    augmented=False. `lmfao inspect <bucket> --episodes` shows the flags.

# 4. TRANSFER (NUC → CUDA pod) — resumable rsync, see
#    docs/policytraining_v3.md §"Data moved" for the pod mechanics:
rsync -rt --partial --append-verify \
  ~/LMFAO/so101/datasets/eval_buckets_v4/ \
  <pod>:/workspace/data/eval_buckets_v4/

# 5. TRAIN (pod) — identical recipe across buckets:
OUT_ROOT=/workspace/data/eval_buckets_v4 \
BUCKETS="stock v2_steady v2_jitter_occlusion" \
./scripts/train_eval_buckets.sh

# 6. RETAIN/UPLOAD — policy roots + datasets per policytraining_v3.md
#    "Retain" steps (HF token passed in-memory from the Mac).

# 7. EVAL (NUC) — load each policy with so101/eval/load_policy.sh, then
#    15–20 trials each:  POLICY=<bucket> ./so101/series 15
#    Score with ./so101/eval_py review.py <run-dir>; audit failures
#    frame-by-frame (track the puck, not the arm) and run
#    ./so101/eval_py eval/diagnose_close.py for the close-pose scatter.
```

Optional, any time before step 1: the free `NSTEP=25` vs `NSTEP=100` A/B on
the existing stock policy (see "Inference lever" above).
