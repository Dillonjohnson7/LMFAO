# V2 PIPELINE — wrist-cam re-record + retrain (the structural fix)

**Why:** the front cam has no parallax at the grasp point — a 1 cm lateral miss is
geometrically invisible, and ±1–2 cm lateral scatter is exactly the measured failure
(runs #13–15, CHECKPOINT.md §3). A wrist camera makes the error observable; new
grasp/place-focused demos re-fill the data well the loss plateau proved empty.
**A new observation space means the old 30 demos cannot be mixed in** — the v2
dataset stands alone, so record a full session (30–40 demos), not a top-up.

Everything below is built and tested end-to-end (writer → 2-cam training →
2-cam inference) on a synthetic dataset; only the physical camera steps and the
paid run remain. Est. cost: **~$2.50–3.50** of the ~$6.33 RunPod balance.

---

## 0. Prerequisites
- The ALAN sentinel containers own the arm + front cam while they run. Check
  `docker ps`; stop them and KEEP them stopped for the whole session:
  `docker stop so101-camera so101-sentinel-web`
- Item-1 outcome (2026-07-20, runs #16–20: 0/5): **NSTEP=50 confirmed clean** (no churn).
- **v2 is evaluated CLEAN (user decision 07-21): no auto-retry, no jam detection, no
  offset tricks.** That apparatus was an item-1 band-aid for v1's blindness and never
  fired correctly; v2's thesis is that data + the wrist cam fix the aim. `~/creep go2`
  = NSTEP=50, front+wrist, and the basic safety rails only (deg/tick clamp, demo
  envelope, start-pose gate, time cap — those stay: they protect the hardware, not
  the metrics). Grip verdicts still print as passive scoring telemetry.

## 1. Cameras (one-time, ~30 min)
1. **Rigidly mount BOTH cameras before recording a single frame** — the front
   mount has drifted twice; drift after recording poisons the dataset the way
   the night-lighting mismatch did (Mistake #10).
2. **The wrist cam is a repurposed Anvil camera** (decision 2026-07-20). Unplug it
   from its current port — its tile on the 8088 viewer goes offline; that's expected
   and permanent. Mount it on the wrist/gripper (view: the jaws + the table in front
   of them), plug it into a **direct** PC USB port (hub lesson, STATUS.md). If it
   re-enumerates on a node from the 8088 list (video0/2/4/6), `--find` will warn —
   answering YES is correct in that case. Then pin it:
   ```
   python /home/anvil/SO101_policy/eval/cam_check.py --find
   ```
   It waits for the new device, warns if you picked a node Anvil's 8088 streamer
   reads (video0/2/4/6), test-captures, and pins the stable by-path name into
   `wrist_cam.path`. Every v2 tool reads that file — no other config to touch.
3. Verify both cams + views: `python /home/anvil/SO101_policy/eval/cam_check.py`
   → check `eval/cam_probe/front.jpg` + `wrist.jpg` (wrist should see the jaws
   + the table in front of them).

## 1.5 State after the 07-21 pipeline validation (nothing to clear)
The whole pipeline was validated end-to-end on 07-21 with the first 2 demos
(CHECKPOINT §2 "v2mini"). Everything is kept:
- `datasets/pick_place_v2` holds eps 0–1 — the final session **APPENDS** to it
  (`~/rec 33`; the recorder resumes automatically). The 2 demos are good data.
- ~~`training/run_v2/checkpoints/last/pretrained_model` currently holds the
  400-step SCAFFOLD~~ **07-22: no longer.** The scaffold sat exactly where the
  watchdog's checkpoint mirror + DONE upload write through, and silently broke
  both (CHECKPOINT Mistake #18). It now lives in `run_v2/v2mini_scaffold_last/`;
  `checkpoints/last` is a symlink rsync keeps pointed at the newest real synced
  checkpoint — `dry2`/`go2` always load the newest real v2 policy.
- `training/run_v2mini/` = validation evidence (train.log, loss_overlay.png).
- After the final session's Hub push, spot-check the repo file list for stale
  files from the 2-demo upload (upload_folder never deletes).
Before recording: `python eval/snap_check.py` (MANDATORY — the wrist carries a
cable) and `python eval/cam_check.py`. Also: `training/rp_key` must be restored
before the cloud run (key was lost in the 07-20 split; RunPod → Settings → API Keys).

## 2. Record the v2 demos (~1 afternoon)
```
~/rec 35
```
The recorder now records **front + wrist** into `datasets/pick_place_v2` and
prompts the protocol per demo (all four levers baked in):
- puck position cycles 9 zones of the demo region (coverage, not luck)
- close SLOWLY — extra frames in the last 3 cm is supervision where it fails
- release at the bin CENTER (run #13 died at the rim)
- every 6th demo = correction demo (start ~1–2 cm off, visibly correct, grasp)

Start every episode from the NEW v2 home pose — the arm base was rotated on
07-21, so v1's start pose is obsolete. The session's ACTUAL demo-mean start
(mean of all 45 recorded first frames — the `[7.0, …]` pre-session estimate
was stale): `[-40.1, -100.6, 96.5, 72.4, -4.8, 1.4]`
(pan/lift/elbow/wrist_flex/roll/grip; read any time with `python eval/read_pose.py`).
Consistency across episodes is what matters — the recorded demos define the pose
gate that `~/creep` will use.
Spot-check a few episodes (side-by-side front|wrist render):
```
python /home/anvil/SO101_policy/view_episode.py 0
```
Sessions resume — `~/rec 10` tomorrow appends to the same dataset. A guard
refuses to mix camera sets in one dataset.

## 3. Push to the Hub
```
python /home/anvil/SO101_policy/training/push_dataset.py
```
(→ private `Dillonjohnson/pick_place_v2`, tag v3.0; canonical meta/data/videos only.)

## 4. FREE smoke test before renting anything (Mistake #11)
```
bash /home/anvil/SO101_policy/training/smoke_test.sh
```
~1 min on the workcell CPU; proves the exact cloud config (2 cams + augmentation)
against the REAL v2 dataset. Do not rent a box until this passes.

## 5. Cloud run (same proven pattern as the 150k run)
```
python /home/anvil/SO101_policy/training/rp.py deploy   # A5000 secure, ~$0.27/hr
```
Then on the box (copy `cloud_setup.sh` + `train_act.sh` over, as before):
```
bash cloud_setup.sh          # deps + CUDA health gate + hf auth (needs HF_TOKEN)
bash train_act.sh            # v2 defaults: pick_place_v2 · 120k steps · AUG on
```
120k not 150k: the plateau was measured flat past ~100k. Two 720p streams cost
~1.5–2× per step → expect **~8–12 h on the A5000 ≈ $2.20–3.30**. Checkpoints
every 10k. Save the model off-box BEFORE terminating:
```
hf upload Dillonjohnson/act_pick_place ~/outputs/train/act_pick_place_v2/checkpoints/last/pretrained_model ckpt-v2-last --repo-type=model --private
```
```
python /home/anvil/SO101_policy/training/rp.py terminate
```

## 6. Pull the model to the workcell
**07-22: OBSOLETE for the live run** — the watchdog mirrors every checkpoint to
`training/run_v2/checkpoints/` and `last` is a symlink tracking the newest, so
`~/creep dry2/go2` load the final model with NO manual step once training
completes. The Hub `ckpt-v2-last` copy (auto-uploaded on DONE) is the OFF-SITE
backup; the commands below are only the recovery path if the local mirror is lost:
```
hf download Dillonjohnson/act_pick_place --include "ckpt-v2-last/*" --local-dir /home/anvil/SO101_policy/training/run_v2/checkpoints/last/_dl
mv /home/anvil/SO101_policy/training/run_v2/checkpoints/last/_dl/ckpt-v2-last /home/anvil/SO101_policy/training/run_v2/checkpoints/last/pretrained_model
```

## 7. Eval ladder (same discipline that caught the loading bug)

**Step 0 — ✅ DONE (07-21/07-22), kept for the record:** `creep_test.py`'s
`TARGETS`/`demo_start` now carry the measured v2 demo-mean start
`[-40.1, -100.6, 96.5, 72.4, -4.8, 1.4]` (updated 07-21 from all 45 first
frames), and `ring_spot.py` was rewritten 07-22: front cam from the pin file
(the old hardcoded Sonix is the WRIST cam now), TARGET **(756,584)** derived
from the 36 clean-pick demo starts, detector re-tuned + size/circularity gated
(validated 36/36 on demo frames). Scale note: ~6.2 px/cm on the v2 front cam.
**Step 0.5 — lighting/scene gate (added 07-22):** run
`python eval/scene_check/refresh.py` and compare live vs training rows for BOTH
cams. Daylight gives the front cam a strong blue cast the demos never saw
(B/R 1.27 at 10:40 vs 0.94 in the demos) — roll out in demo-like evening light,
and clear the table of anything not in the demos (phone/cables seen there 07-22).
1. Offline, training-parity check first:
   ```
   MODEL=/home/anvil/SO101_policy/training/run_v2/checkpoints/last/pretrained_model DATASET_ROOT=/home/anvil/SO101_policy/datasets/pick_place_v2 REPO=local/pick_place_v2 /home/anvil/SO101_policy/.venv/bin/python /home/anvil/SO101_policy/eval/offline_eval.py
   ```
   Expect a few degrees MAE; ~1.0 *normalized* error = random net — stop and check.
2. `~/creep dry2` — grounding + one inference on the live scene (also shows the
   2-cam CPU inference time; was 0.34 s with one cam, expect roughly double —
   still fine inside a 50-tick replan window).
3. Place the puck with the aimer — placement error wrecked 3 of 5 runs on 07-20:
   ```
   /home/anvil/SO101_policy/.venv/bin/python /home/anvil/SO101_policy/eval/ring_spot.py
   ```
   (repeat until `ON TARGET - go`; also use it to place at deliberate offsets when
   testing whether v2 actually tracks off-mean pucks — the localization question)
4. `~/creep go2 135` — CLEAN rollout: NSTEP=50, front+wrist, basic safety rails,
   **no retry/jam heuristics** — the policy sinks or swims on its own. Score:
   grip <20 = air, 26–36 = held + LIFTED; footage in `eval/run_recordings/`;
   update CHECKPOINT.md.

---
**OUTCOME (2026-07-22 night): the pipeline delivered.** First-ever full task on rollout #1
(run #21), 4/13 over the night, NSTEP=100 the winning config (bracket in CHECKPOINT §0/§3).
This doc is now HISTORICAL — the v2 system is live; current state and next steps live in
CHECKPOINT.md (§0 status + footer's morning pickup list).
