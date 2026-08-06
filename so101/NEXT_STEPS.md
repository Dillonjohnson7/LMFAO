# NEXT STEPS — the active two-item plan (distilled 2026-07-20)

> **OUTCOME 2026-07-20 late:** Item 1 ran (runs #16–20): **0/5 grasps** — NSTEP=50 is clean
> (no churn) but the close still misses ±1–2 cm; the retry's jam detector needed two live
> calibrations (see CHECKPOINT §3 Era 3). Policy declared insufficient as-trained.
> **→ ACTIVE: Item 2, `V2_PIPELINE.md`.** Wrist cam = a repurposed Anvil camera; full re-record.

**Context:** 150k ACT, first verified pick (run #13, place missed ~2 cm), series 1/3
grasps. Measured bottleneck: **±1–2 cm model scatter on the final approach** (NOT camera
bias — misses went right-rear in #14, left-near in #15) and the misses are **correlated
within a run** — the policy commits to a wrong estimate it cannot see. Evidence: `CHECKPOINT.md` §3.

## ITEM 1 — short-term, $0, built & selftested, ready to run
Both changes attack the measured bottleneck with the current 150k model. In order:
1. **NSTEP=50 series** (evidence-bracketed: 10 churned the close in run #9, 100 pecked
   blind in #14–15): `~/creep go50 135` ×3, puck at the demo-mean spot. Fall back to
   `go30` only if 50 re-introduces churn.
2. **Close-on-air auto-retry** on the winner: `~/creep retry 135` ×3. Detector = grip <20
   sustained after an open (called #13–15 correctly); action = guard-railed re-home to
   demo start with **alternating pan offsets** (the offset breaks the within-run miss
   correlation; unperturbed retries repeat the same miss). Max 3 per run.
Scoring: grip <20 = air, 26–36 = held (the harness now prints these verdicts live);
footage in `eval/run_recordings/`; update CHECKPOINT.md §3.
`~/creep check` = safe no-robot selftest, run any time. If the sentinel containers are
up, stop them first (the wrapper warns).

## ITEM 2 — the v2 retrain package (wrist cam + focused demos + augmentation)
**Runbook: `V2_PIPELINE.md`** — cameras → `~/rec 35` (protocol baked into the prompts) →
push → free smoke test → A5000 run (120k steps, ~$2.50–3.50) → eval ladder → `~/creep go2`.
The whole software path is pre-tested end-to-end on a synthetic 2-cam dataset; only the
physical camera mounting and the paid run remain.

**Dropped from the old 7-option list** (evidence didn't support them): the 100k/150k
checkpoint A/B (offline says equivalent; the memorization theory has no data point —
back-pocket only), the augmentation-only retrain (folded into item 2 as a flag),
pi0/SmolVLA (premature until the observability+data fix is tried), more steps (plateau
measured), more camera calibration (falsified by the miss directions).

---

## The original 7 options (2026-07-18, kept for reference)

---

## 1. Grasp- + place-focused demos → retrain (~$2.30) — THE BIG LEVER
- 15–20 demos emphasizing slow, deliberate closes from varied puck positions
- PLUS a handful of exaggerated place-into-bin-CENTER releases (run #13 released at the rim)
- Rationale: loss plateau proved the 30-demo well is empty; footage shows failure concentrated
  in the last 3 cm of approach. Same pipeline that already works (`~/rec` → Hub → runbook).
- Status: was on hold per user (07-18); revisit when ready to record.

## 2. Add the wrist camera + re-record (~1–2 hrs recording + ~$2.50 retrain) — STRUCTURAL FIX
- The fixed front cam has no parallax at the grasp point — policy can't see a 1 cm left/right miss.
- Hardware already owned: the spare camera on the port-8090 streamer (see so101-cameras memory;
  DO NOT touch Anvil's cams/8088).
- Most likely option to take grasps from 1/3 → near-100%.
- Caveat: new observation space invalidates current dataset → pair with option 1's recording
  session (one combined re-record covers both).

## 3. Behavioral A/B of checkpoints 100k–150k (FREE, ~30 min)
- Offline MAE 100k≈150k (0.9° vs 0.8°) but offline MAE doesn't measure grasp precision;
  extra steps on 30 demos may memorize rather than sharpen.
- All 15 ckpts local: `~/SO101_policy/training/run150k/checkpoints/`
- Run: `MODEL=~/SO101_policy/training/run150k/checkpoints/100000/pretrained_model ~/creep go 135`
  (in-script env export is safe; NEVER rely on typed env prefixes — Mistake #12)
- ~3 scored runs per candidate checkpoint (100k, 120k vs 150k baseline).

## 4. Tune re-plan interval between 10 and 100 (FREE)
- Evidence brackets it: NSTEP=10 churned the close into oscillation (run #9); NSTEP=100
  commits so hard it can't correct a 1 cm miss mid-descent (#14–15 blind pecking).
- Try NSTEP=50 (and maybe 30): keep run-#10 decisiveness, allow one visual correction
  during descent. `NSTEP=50` must be baked/exported in-script, not typed as prefix.
- 2–3 scored runs per setting.

## 5. Retrain with image augmentation, same data (~$2.20, no new demos)
- lerobot image transforms: random crop/shift, color jitter.
- Makes policy robust to the accepted +23 px camera offset + future mount drift;
  effectively multiplies the 30 demos.
- Best retrain available without touching the robot. Won't fix intrinsic scatter as
  well as new demos.

## 6. Grip-triggered retry logic in the harness (FREE, ~1 hr engineering)
- We have a proven close-on-air detector: **grip < 20 after a close** (called #13–15 correctly).
- On detection: re-home to approach pose with a small lateral offset, let policy retry →
  turns the blind peck loop into a systematic search.
- Band-aid, not a policy fix — but run #13 proves ONE good close completes the whole task.
- Implement in `~/SO101_policy/eval/creep_test.py`.

## 7. Fine-tune pi0 or SmolVLA on the same dataset (~$5–15) — MOONSHOT
- Pretrained visuomotor backbone squeezes more precision/generalization from 30 demos
  than ACT-from-scratch.
- Hub dataset + cloud runbook (`~/SO101_policy/training/`) make it mostly a config change.
- Costlier per experiment; slower CPU inference on workcell. Hold until cheap options exhausted.

---

## Recommended sequencing
1. **Free trio first**: #3 + #4 = a couple of scored series this afternoon; #6 = an hour of
   harness work.
2. Then decide **#1 vs #2 in one combined recording session** (both mean re-recording —
   do them together: wrist cam + grasp/place-focused demos).
3. **#5** slots in any time a retrain is wanted without touching the arm.
4. **#7** after the above are exhausted.

Keep the scoring discipline: demo-mean puck spot, `~/creep go 135`, review the self-recorded
footage per run (`eval/run_recordings/`), grip <20 = air / 26–36 = held, update CHECKPOINT.md.
