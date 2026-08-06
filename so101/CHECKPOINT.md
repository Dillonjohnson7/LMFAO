# CHECKPOINT DOC — SO101 pick-and-place policy project

**Written:** 2026-07-18 · **Machine:** anvil-workcell · **Status: V2 WORKS — 07-22 night tally: 4/13 full tasks overall, but at NSTEP=100 it is ~3/6 (runs #29, #31?, #33? OK · #32 NG · plus user-run successes; #31/#33 labels PROVISIONAL — user confirms AM).** NSTEP bracket complete & monotone: N10 freeze (#30) · N30 hover (#28) · N50 shallow bites (1/7, #21–27) · N100 committed closes. v1 for comparison: 1/8 grasps, 0 full tasks, ever. The close still scatters; commitment is the best free mitigation found. Next session: confirm provisional labels → build the clean N100 sample → gripper overload check → decide the retrain (grasp-focused demos, ~$4–5, balance $0.92 → needs top-up).
**2026-07-20:** split out of `~/ALAN` into its own repo, `~/SO101_policy` (own lerobot checkout + `.venv`; the ALAN sentinel stays at `~/ALAN`). All paths below updated.
**2026-07-20 (later):** two-item plan built & pre-tested (see `NEXT_STEPS.md`): item 1 =
NSTEP=50 series + close-on-air auto-retry in the harness (`~/creep go50|retry`, selftest
`~/creep check`); item 2 = wrist-cam v2 re-record/retrain pipeline (`V2_PIPELINE.md`),
validated end-to-end on a synthetic 2-cam dataset (record features → train+aug → infer).
Note: the ALAN sentinel containers were seen RUNNING today — they hold the arm + front
cam; the wrappers now warn. Next CHECKPOINT update: the item-1 scored series (#16+).
This is the full record: every run, every training, every mistake, every lesson.
Read this to understand exactly where we stand and how we got here.

**TERMINOLOGY (settled 2026-07-22, user call): the object is a ~7 cm BLUE PUCK** — a flat
disc, grasped by wrapping the jaws around its outside, carried at grip 26–36. The datasets'
baked-in task string says *"cube"* (recording-time label; cannot change retroactively — the
`TASK` constants in `creep_test.py`/`guided_record.py` must keep matching it) and docs/cards
called it a *"ring"* for a while (07-18→07-22). Cube, ring, and puck in any historical
artifact all refer to this same puck. `ring_spot.py` keeps its historical filename.

---

## 0. Where we stand right now (TL;DR — updated after run #15)

- **RUN #13 PICKED UP THE PUCK.** Grasped (carry grip 31, demo band 26–36), lifted, carried to the
  basket, released ~2 cm short — puck bounced off the near rim onto the table. First verified pick.
- Series score 1/3 grasps, 0/3 full task. Misses are **not** consistently rightward (#14 right-rear,
  #15 left-near) → model scatter, not the camera offset, is the bottleneck. Details in §3.

- **A trained ACT policy exists and is GOOD**: 150k steps, loss 0.058, **0.8° mean action
  error at true training parity** (verified after fixing the harness bug that hid it).
- **BREAKTHROUGH (run #10, native 100-tick commitment):** the policy executed the **entire
  task sequence** — reach → open → descend → close → lift → carry toward bin → release —
  in ~33 s. But the close went to grip **6.2** (demos hold the puck at **26–36**) → it had
  **closed on air** beside the puck; the carry was empty. Runs #11–12 (self-recorded,
  reviewed frame-by-frame): approach + centering reliable every time; the close never
  secures the puck — repeated descend-peck-retry cycles.
- **Current bottleneck: the CLOSE is a ±1 cm precision task** (outside-wrap grasp of a
  ~7 cm puck). Error budget: ~1–2 cm model precision + ~1 cm residual camera bias
  (dx +23 px, user-accepted) + half-speed contact dynamics.
- **2026-07-20 evening — the series ran (#16–20): 0/5 grasps, cumulative 1/8.** NSTEP=50 is
  clean (no churn, committed close) but the close still misses ±1–2 cm — run #19 at a
  *verified* demo-mean placement settles it: the error is invisible to the front cam, so no
  re-plan interval or retry logic fixes the aim. **USER DECISION: stop v1 rollouts. Next
  session runs `V2_PIPELINE.md`** — wrist cam (a repurposed Anvil camera) + full demo
  re-record + retrain. Placement tooling now exists: `eval/ring_spot.py` (live aimer).
- **2026-07-21 — v2 camera/arm bring-up started (in progress).** The arm was **rotated in the
  scene** (base moved, servos untouched): measured `shoulder_pan` **+47.0°** vs the v1 demo-mean
  start with every other joint within ±1.5°, plus a `wrist_roll` **~170° flip** from mounting the
  camera bracket. That signature = pure repositioning, **so no `lerobot-calibrate` was run and
  `follower.json` is unchanged** (backed up to `calib_backup/` anyway). The v1-derived constants
  `creep_test.py:167` (`TARGETS`) and `:213` (`demo_start`) are therefore **stale and not yet
  rewritten** — do that once the home pose is final, and rebuild the min–max envelope from the v2
  demos. `ring_spot.py`'s `(400,460)` target is stale for the same reason. Camera roles **swapped**
  and a third camera-owning project was discovered — see the camera map in §1 and Mistakes #15–17.
  **USER DECISION 07-21: v2 rollouts are evaluated CLEAN — the item-1 auto-retry / jam-detector
  apparatus is dropped from the v2 path** (`go2` = NSTEP=50 + safety rails only). Rationale: it
  was a band-aid for v1's blindness, needed two live calibrations, and never fired correctly
  (run #20); v2's bet is data + observability, and the eval must measure that bet unconfounded.
  The retry code remains available for the v1 modes (`retry`/`retry30`) only.
  **LEADER RE-ZEROED 07-21 16:35 (user call — no arm was moved):** teleop connect snaps the
  follower to the leader's *reading*, and the leader still read the v1 pose (pan −51.8°, roll
  +169.7° off the follower) — a connect would have whipped the wrist and torn the new camera
  cable (`eval/snap_check.py` caught it). Fix: `eval/match_leader.py` rewrote the leader's
  Homing_Offset registers so current-physical reads == follower's pose (slope measured
  empirically, every write read back and verified; wrist_roll needed a −4096 sign-magnitude
  fold, proven by read-back). Worst residual 0.1°. `leader.json` updated in cache + repo;
  pre-change backup in `calib_backup/`. ~~⚠ `~/ALAN`'s leader.json copy is now STALE~~ —
  synced 2026-07-22. **Run `eval/snap_check.py` before EVERY teleop/`~/rec` session now
  that the wrist carries a cable.**
- **2026-07-21 late evening — V2 DATASET COMPLETE, 150k TRAINING LAUNCHED (overnight).**
  - **Recorder bug found & fixed mid-session:** buffered ENTERs (key-repeat / presses during
    the multi-second `save_episode()` encode) auto-answered the next prompt — skipping STEP 4
    and, worse, once auto-KEEPING a junk episode (ep3: 120 frames, zero joint motion —
    verified from footage + parquet, deleted via `lerobot-edit-dataset`, backup in
    `pick_place_v2_old/`). Fix: `termios.tcflush` before every prompt in `guided_record.py`,
    pty-tested. Side effect of the delete: eps 0-2's videos re-encoded near-all-intra AV1
    (~5x bulkier; decode fine; optional `reencode_videos` later).
  - **Final dataset: 45 episodes, 31,063 frames** (Hub `pick_place_v2` tag v3.0, mirror-pushed).
    Composition: eps 0-35 clean picks · 36-40 rim-recovery · 41-42 mid-descent displacement ·
    43-44 low-light (brightness 113 vs 170 — measured). Health screen: no junk, gripper action
    everywhere, scene geometry identical across the session (frame-compared eps 0/20/40).
    Start poses drift sigma=7.6 deg in pan (operator variance; harmless variety). Known wart:
    ep28 video window carries ~2.5 s of redo-orphan frames (Mistake #16; training unaffected).
    NOTE: V2_PIPELINE's "[7.0, ...]" home pose is STALE — the session's real demo-mean start is
    `[-40.1, -100.9, 96.5, 72.4, -4.8, 1.4]`; `creep_test.py` TARGETS/demo_start updated to it.
  - **Training: STEPS=150000 (USER DECISION — more data than v1), AUG on, batch 8, ckpt/10k,**
    on a SECURE **RTX A4500 $0.25/hr** (A5000 out of stock; `launch.sh` now walks a GPU
    fallback list). Pod `zpbsn9ltcv2cvb`. Est. 13-18 h ≈ $3.50-4.50. Smoke test passed first;
    CUDA pre+post gates green. Watchdog caps (USER DECISION): alert $4.50 / hard-kill $5.75
    (kill uploads the mirrored ckpt as `ckpt-v2-costcap` first). On DONE: auto-upload
    `ckpt-v2-last` + auto-terminate. Monitoring: rebuilt rerunnable stack in
    `training/run_v2/` (dashboard :8095 · watchdog · remote_run.sh · launch.sh) — all
    repo-local paths, parser fixture-tested on both v1 GPU and v2mini CPU log formats.
  - **RunPod pipeline hardened:** new API key in `training/rp_key` (chmod 600, gitignored);
    `rp.py` defaults SECURE+A5000; `push_dataset.py` mirror-pushes (`delete_patterns`);
    `train_act.sh` upload hint now includes `ckpt-v2-last`; `cloud_setup.sh` gained the
    image-torch CUDA pre-gate (Mistake #4, institutionalized).
- **2026-07-22 morning — mid-training check + rollout prep (training at 52%, on track).**
  - Step 79k/150k, loss 0.075 (tracking slightly AHEAD of the v1 curve at the same step),
    2.00 steps/s steady, GPU 100%/77°C. Revised ETA ~20:20 tonight, ~21 h ≈ **$5.30 total** —
    the $4.50 alert line will fire (~17:00, expected, ignore); ~$0.45 margin under the $5.75
    hard-kill (which was raised for exactly this case).
  - **WATCHDOG UPLOAD BUG found & fixed (would have shipped the wrong model tonight):** the
    pod's `checkpoints/last` is a SYMLINK; rsync could not replace the local v2mini-scaffold
    DIRECTORY with it (`cannot delete non-empty directory` every cycle, Mistake #18) — so the
    DONE path would have uploaded the **scaffold** to the Hub as `ckpt-v2-last` (same trap in
    the costcap path). Fix: scaffold moved to `run_v2/v2mini_scaffold_last/`, `last` is now a
    local symlink rsync keeps pointed at the newest synced checkpoint. Side effects: the
    `~/creep dry2/go2` MODEL path now always resolves to the newest real v2 checkpoint, and
    the footer's "pull the ckpt from the Hub" step is OBSOLETE — the local mirror is canonical.
  - **Rollout prep done against the synced 70k checkpoint (all offline/read-only):**
    go2-config selftest ALL PASS (2-cam inference 1.05 s/chunk CPU); offline eval MAE **1.45°**
    overall (healthy mid-training; v1 finished at 0.8°) — the full v2 eval path is proven.
  - **`ring_spot.py` rewritten for v2:** front cam now read from the pin file (it pointed at
    the Sonix = today's WRIST cam); target re-derived from frame 0 of all 36 clean-pick demos:
    **TARGET (756,584)**, σ (31,26) px, puck r≈22 px → **~6.2 px/cm** (v1 was 15.7 — the new
    front cam sees the puck 2.6× smaller, so pixel tolerances shrank accordingly). Detector
    re-tuned (the Anvil 4K desaturates the puck; S floor 120→25) + size/circularity gates —
    validated 36/36 on demo frames AND rejects daylight blue-cast false positives (live-tested).
    Overlay: `eval/cam_probe/v2_ring_targets.jpg`.
  - **Mistake-#1 guard INSTITUTIONALIZED (user call, after dry2):** `creep_test.py` now
    byte-verifies EVERY tensor of the loaded policy against `model.safetensors` on every
    run (~1.5 s; hard-abort on mismatch; prints the resolved checkpoint realpath into the
    run log). Standalone: `eval/verify_weights.py`. The checker itself was negative-tested:
    it correctly FAILs the 80k policy against the scaffold's file (153/234 tensors differ).
    Positive proof at 11:30: dry2's policy = the 080000 mirror, 51,668,614 params identical.
  - **`eval/scene_check/` rebuilt for v2 (2 rows: front + wrist)** with a real generator,
    `eval/scene_check/refresh.py` (v1 page was static ad-hoc jpgs). Wrist cam verified live:
    jaws + table in frame, sharp. ⚠ Pre-go2 gate tonight: re-run refresh.py and compare —
    at 10:40 daylight the front cam ran B/R 1.27 vs the demos' 0.94 (strong blue cast);
    evening light should match, but CHECK before rolling.
- **2026-07-22 20:24 — V2 TRAINING COMPLETE, EVERYTHING VERIFIED. Ready for rollouts.**
  - **150,000 steps, final loss 0.056** (v1 finished 0.058 on the easier 1-cam task; the v2
    curve ran ahead of v1's the whole way). 21h07m @ 2.00 steps/s on the A4500. **$5.40**
    (margin to the $5.75 kill: $0.35). Watchdog DONE path worked end-to-end: final sync →
    `ckpt-v2-last` Hub upload → pod terminate — each step INDEPENDENTLY verified (no pods
    billing, balance $0.92; Hub file sizes match local byte-counts; `last -> 150000`).
  - **Model gates all green:** weights byte-verified 234/234 tensors (51,668,614 params) via
    the new every-run guard; **offline parity MAE 1.09°** overall on the final ckpt (was
    1.45° at 70k; v1's 150k measured 0.8° same-style) — wrist_roll 0.03°, gripper 0.63°.
  - Remaining tonight = the live ladder: scene refresh (evening light!) → `~/creep pose` →
    `ring_spot.py` puck placement → `~/creep dry2` → `~/creep go2 135` CLEAN series.
- **2026-07-23 morning — loss-curve analysis + chart suite (no robot work yet).** Four
  renderings of both runs' full 750-pt logs now live in `training/`:
  `loss_v1_vs_v2.png` (clean/annotated) · `_technical.png` (TensorBoard-style + run-spec
  panel) · `_pytorch.png` (4-panel default-matplotlib training report) · `_focus.png`
  (log EMA curve + final-50k linear inset — the keeper). Analytics extracted while
  building them: **v2's loss ≤ v1's at every matched step** (0.093 vs 0.104 @50k ·
  0.070 vs 0.073 @100k · 0.056 vs 0.058 @150k) despite aug ON, a 2nd camera, and fewer
  epochs seen (38.6 vs 56.6 — so the lead is understated, not flattered); tail slopes
  over the final 30k: v1 −0.0025, v2 −0.0022 per 10k — both ended in the same measured
  plateau regime, confirming 150k was the right stop for v2 too.
- Total cloud spend: **~$3.67** of $10 (balance ≈ $6.33; v2 run bills from $6.32 baseline).
- Biggest event of the project remains: **the silent weight-loading bug** — every eval and
  rollout before 2026-07-18 midday ran RANDOMLY INITIALIZED networks; all conclusions from
  that era were re-done.

---

## 1. The asset inventory (what exists, where)

### Data
| asset | location | notes |
|---|---|---|
| Dataset (30 demos, 21,186 frames, 30fps, LeRobot v3.0) | `~/SO101_policy/datasets/pick_place` + **HF `Dillonjohnson/pick_place`** (private, tag `v3.0`) | 457.9 MB canonical (`meta/data/videos`); local `images/` (4.1 GB) is recorder scratch, excluded |
| Demo start pose (mean of 30) | — | `pan -40.2, lift -100.9, elbow 95.3, wrist_flex 73.8, roll 165.0, grip 3.5` (±~5°) |
| Puck start positions across demos | measured 2026-07-18 | varied over **190×247 px** region (x 253–443, y 400–647 in camera), detected 30/30 |

### Models
| asset | location | notes |
|---|---|---|
| **150k ACT (the real model)** | `~/SO101_policy/training/run150k/checkpoints/last/pretrained_model` + Hub `Dillonjohnson/act_pick_place/ckpt-150k-last/` | loss 0.058 · offline MAE **0.8°** (normalized 0.058 = exact training parity) |
| v2mini scaffold (NOT a real policy) | `training/run_v2/v2mini_scaffold_last/` + Hub `…/ckpt-v2mini-last/` | 400-step CPU validation artifact (07-21). Moved OUT of `checkpoints/last` 07-22 (it was blocking the sync AND the DONE upload — see §0); `checkpoints/last` is now a symlink to the newest synced real checkpoint. |
| v2 dataset (FINAL: 45 eps, 31,063 frames) | `datasets/pick_place_v2` + **Hub `Dillonjohnson/pick_place_v2`** (private, tag v3.0) | eps 0–35 clean picks · 36–40 rim-recovery · 41–42 displacement · 43–44 low-light. Review mp4s: `datasets/pick_place_v2/_review/` |
| Loss-chart suite (v1 vs v2, 4 renderings) | `training/loss_v1_vs_v2{,_technical,_pytorch,_focus}.png` | built 07-23 from both train.logs (750 pts each); `_focus` = log EMA + final-50k inset; regenerate any variant from the logs in seconds |
| All checkpoints 10k–150k (15) | `~/SO101_policy/training/run150k/checkpoints/` | 100k ckpt: 0.9° / 0.074 — plateau confirmed behaviorally too |
| 20k ACT (proof-run model) | `~/SO101_policy/models/act_pick_place` + Hub repo root | superseded; **Hub model card still carries INVALID eval conclusions (random-net era) — needs correcting** |

### Tooling
| asset | what it does |
|---|---|
| `~/creep` → `~/SO101_policy/eval/creep_test.py` | rollout harness. Modes: pose (limp + live joint readout, ENTER locks) / dry (grounding check + one inference, no motion) / go (guard-railed rollout). Env: `FAST=1` (2× clamps, NSTEP 25), `NSTEP=`, `DURATION`, `MODEL=`. Logs to `~/SO101_policy/eval/creep_runs.log` |
| `~/SO101_policy/training/` | `cloud_setup.sh` (GPU-box bootstrap: pinned lerobot+[training], CUDA gate, HF auth; `NOVAE=1` supported), `train_act.sh` |
| `~/SO101_policy/training/rp.py` | RunPod API driver (deploy/status/list/waitssh/terminate). Key from `$RUNPOD_API_KEY` or gitignored `training/rp_key`. Needs `User-Agent` header (Cloudflare 1010). *(Preserved 2026-07-18 from the ephemeral scratchpad it used to live in.)* |
| `~/SO101_policy/training/run150k/` | full 750-pt `train.log`, `gpu.log`, `watchdog.sh` (10-min sync, cost caps, auto-terminate), `dashboard.py` (**:8095** — HP bar + live curve, all-real numbers), final curve PNGs. *(watchdog/dashboard are ARCHIVES of the completed run — they reference a scratchpad now gone; use `training/rp.py` for new runs.)* |
| `~/SO101_policy/eval/scene_check/index.html` | camera-vs-training comparison page (live + 5 random training frames) |
| `~/SO101_policy/eval/offline_eval.py` + scratchpad `eval_ckpts.py`, `latent_test.py` | offline eval (`python offline_eval.py [n_frames]`, evaluates the start of the data) / checkpoint compare / VAE-latent A-B test |
| `~/SO101_policy/eval/read_pose.py` | **READ-ONLY** joint-angle readout (`python eval/read_pose.py [secs]`), printed in the same units as `TARGETS`/`demo_start` so it pastes straight in. Never writes, never touches torque — safe while limp *or* while holding. Talks to the bus directly on purpose: `SOFollower.connect()` runs `configure()`, whose `bus.torque_disabled()` ctx manager **guarantees torque is re-enabled on exit**, so "just connecting to read" silently stiffens a limp arm. Added 07-21 to re-measure the home pose after the base rotation. |
| ~~`relax.py`~~ | **BROKEN since the 07-20 split** — imports `sentinel_teleop`, which stayed at `~/ALAN` (Mistake #16). Use `~/creep pose` instead. |
| `~/SO101_policy/view_episode.py` | render one dataset episode to a Firefox-playable H.264 mp4 for demo review (`python view_episode.py [episode_index]` → `datasets/pick_place/_review/`). The only demo-episode viewer — used when reviewing recorded demos. |

### Environment facts
- Workcell: 12-core x86, **no GPU** → training in cloud, inference on CPU (0.34 s/chunk — fine).
- lerobot **0.6.1 not on PyPI** → pin `git+…@3f2179f3b69708b6ad009b2e7685dd9d05269ee1` everywhere.
- The ALAN/sentinel project (separate!) owns the arm+camera when its containers run — they are
  currently **stopped**; restart is a user decision.

### Camera map (measured 2026-07-21) — THREE projects share this box
A **third** project, the Anvil workcell **ROS2 stack** (`anvil-loader-ros2-1`, privileged),
owns four cameras and streams them with one `usb_cam_node_exe` each at 1920x1080 MJPEG.
udev (`/etc/udev/rules.d/99-camera.rules`) pins them **by PCI address** and publishes the map
as symlinks in `/run/cameras/`:

| symlink | PCI slot | node (07-21) | status |
|---|---|---|---|
| `cam_waist` | 0000:03:00.0 | `/dev/video2` | ROS2 stack, streaming — **do not take** |
| `cam_chest` | 0000:04:00.0 | `/dev/video4` | ROS2 stack, streaming — **do not take** |
| `cam_wrist_r` | 0000:05:00.0 | `/dev/video6` | ROS2 stack, streaming — **do not take** |
| `cam_wrist_l` | 0000:06:00.0 | — | **unplugged 07-21 15:00 — repurposed for SO101** |

SO101's v2 cameras, settled 07-21 (~16:10) after some churn:
- **WRIST = the Sonix `USB2.0_CAM1`** (`05a3:9230`) — the v1 front cam, physically remounted
  onto the wrist WITHOUT unplugging its USB (stayed at hub port `1-7.3` the whole time).
  `wrist_cam.path` → `…usb-0:7.3:1.0-video-index0`. Verified: 30 fps, jaws + table view.
- **FRONT = a repurposed Anvil 4K** (`1bcf:2d4f`, no serial) on the **`cam_wrist_r` cable**,
  PCI `0000:05:00.0`. `front_cam.path` → `…pci-0000:05:00.0-usb-0:1:1.0-video-index0`.
  Verified: 30 fps, sharp scene view. NOTE: the FIRST repurposed unit (ex-`cam_wrist_l`) was
  rejected — its fixed focus was wrong for the scene distance; a second Anvil unit replaced it
  (these no-serial units are interchangeable to udev, which pins the PORT, not the camera).
  The Anvil stack is left with only `cam_chest` connected; the `cam_waist` port sits empty.
`cam_check.py` pins both roles explicitly (`wrist_cam.path`, `front_cam.path`, via
`--find`/`--find-front`/`--set`/`--set-front`). `eval/live_view.py` (added 07-21) serves both
cams live on **:8096** with a Laplacian-variance sharpness score for focusing/aiming.

⚠ **Never plug an SO101 camera onto `0000:06:00.0`** — udev would recreate
`/run/cameras/cam_wrist_l` and the next `docker restart` of the ROS2 container would reclaim it
mid-session. Use a motherboard port, which no camera rule matches.

⚠ **The wrist cam sits on the culprit hub** (user-accepted 07-21). `/dev/video0` is at `1-7.3`
on hub `1a40:0101` Terminus, alongside the two `canable2 gs_usb` CAN adapters — the same hub
that caused the follower USB-drop disaster (§4 / STATUS.md). It has been stable there for the
whole project, but the cable now flexes with the wrist. If wrist frames ever drop or the fps
sags mid-session: check `journalctl -k | grep -i "disabled by hub"` FIRST, and the fix is a
direct motherboard port (re-pin with `--set` after moving).

⚠ **USB continuity ≠ mounting continuity** (07-21, burned a round of confusion): a camera can be
remounted anywhere without a replug, so neither the kernel log nor the image alone identifies
its role — with the arm rotated +47°, a fixed front cam frames just gripper+table, which looks
exactly like a wrist view. Confirm by moving one joint a few degrees: background moves = fixed
cam; jaws stay put = wrist cam. Or ask the human who mounted it.

---

## 2. Training ledger

| run | hardware | config | result | cost |
|---|---|---|---|---|
| **20k proof** (07-17) | RunPod **A100 80GB PCIe** secure, $1.39/hr (after a **dud community 4090** — CUDA err 999 with healthy nvidia-smi) | batch 8, lr 1e-5, pyav | 49 min @ 7.3 steps/s; loss 2.11→**0.167**; pipeline proven end-to-end | ~$1.46 (incl. dud + retries) |
| **v2mini scaffold** (07-21) | **workcell CPU** (RunPod leg BLOCKED: API key lost with the pre-split /tmp scratchpad — `training/rp_key` needs restoring) | v2 config: 2 cams + AUG, batch 2, **400 steps**, ~7.3 s/step | loss 77→**3.61** — overlay vs the 150k curve: `training/run_v2mini/loss_overlay.png` (same early power-law shape; same loss neighborhood at overlapping steps 200–400); purpose = END-TO-END v2 PIPELINE VALIDATION with the first 2 demos, not a usable policy. Validated: Hub push (`pick_place_v2` v3.0) → **Hub pull-train** (private auth, fresh download, losses identical to local) → checkpoint → `hf upload` (`ckpt-v2mini-last`) → `hf download` to the `run_v2/.../last/pretrained_model` path → offline eval (MAE 12.1° — underfit as expected, NOT the random-net signature; Mistake #1 discipline) → **live 2-cam inference** off the real pinned cameras (finite actions, envelope clamp engaged, **1.11 s/chunk fresh, ~3 ms queued** — v1 was 0.34 s; watch this on the real v2 model) | **$0.00** |
| **v2 150k full** (07-21→22, overnight+day) | RunPod **RTX A4500** secure $0.25/hr (A5000 out of stock) | 2 cams + AUG, batch 8, STEPS=150000, ckpt/10k | **21h07m** @ 2.00 steps/s; loss 77→0.075@78k→**0.056@150k** (beat v1's same-step loss throughout, despite 2 cams); watchdog DONE path: mirror + Hub `ckpt-v2-last` + auto-terminate, all verified; offline MAE **1.09°** | **$5.40** |
| **150k full** (07-17→18, overnight) | RunPod **A5000** secure $0.27/hr, 96-core host | same, STEPS=150000, ckpt every 10k | **7h53m** @ 5.37 steps/s, GPU 95% util; loss **0.251@10k** (reproduced 20k-run curve exactly) → 0.165@20k → 0.104@50k → 0.073@100k → **0.058@150k**; plateau measured: slope −0.002…−0.004 per 10k past 100k | **$2.21** |

Curve model: power-law `loss ≈ 624·step^-0.827` fit the early curve, but the tail plateaued
**~75% above** its extrapolation (predicted 0.033@150k, got 0.058). Lesson: extrapolations
gave the right *shape* and wrong *floor*; past-plateau steps buy nothing — **the 30-demo well
is fully pumped.** Monitoring stack (watchdog + dashboard + cost caps + auto-terminate) worked;
the pod killed itself on completion.

---

## 3. Rollout ledger (`creep_runs.log` + earlier un-logged runs)

**Era 1 — random networks (INVALID, see Mistake #1):** 3 rollouts of "20k" (07-17) and 2 of
"150k" (07-18 morning): all showed the same "rise to blurred mean pose, hover, ignore puck"
behavior — including a puck-removal test. All of it was the behavior of untrained networks.

**Era 2 — the real 150k model (07-18, after the fix):**

| # | config | puck | what happened |
|---|---|---|---|
| 3 | creep (NSTEP 10 eff., 45 s) | normal spot | first-move prediction 3.2° ("stay at start" — correct); smooth demo-shaped reach; **gripper began opening at t=44 — 1 s before timeout** |
| 4 | 135 s | normal | reached, then **fenced 5–8° above the puck for 80 s** → found the envelope clamp bug (Mistake #8) |
| 5 | 135 s | moved | **declined to leave start pose for 40 s** (scene-dependent behavior — first vision evidence), then python segfault (core dump; faulthandler added) |
| 6 | 60 s, envelope fixed | moved | reached the *demo-mean* spot (not the puck), hovered |
| 7 | — | moved | froze at start 46 s |
| 8 | intended FAST — **actually ran 10-tick** (env prefixes never reached the process; log line proves it) | normal | reach → open → descend → jaws ON the puck (photo) → held open at grip 41 |
| 9 | same (10-tick) | normal | stalled mid-close, grip oscillating 15–18 — close being churned by re-planning |
| **10** | **native: FAST + NSTEP=100** (baked into `~/creep go` as single token after the env-prefix failure) | normal | **★ FULL SEQUENCE**: reach (3.9 s!) → open → descend → close (t=22) → lift/retract → carry toward bin (pan+) → release (t=30). BUT closed to grip **6.2** vs demos' carry-grip **26–36** → **closed on air**; puck ended back on table |
| 11–12 | native, **self-recorded** (robot-view frames, new harness feature) | normal | approach + centering excellent both runs; **close never secures the puck** — repeated descend-peck-retry cycles (learned retry behavior?); puck never left the table |
| **13** | native, self-recorded (13:12) | demo-mean | **★★ FIRST REAL PICK.** Close at t≈10 caught the puck body (carry grip 31→30, dead in the 26–36 demo band); clean lift + carry to the basket (airborne in frames t=12.5–14.6); **released ~2 cm short at t≈16 — puck landed on the near rim and bounced outside** onto the table. Policy then re-approached the puck at its new spot (OOD position!), closed on air (grip 9.6), returned to start. Task ~90%: failed only at the place. |
| 14 | native, self-recorded (13:13) | demo-mean (nearer/left) | grasp FAIL — whole run one peck-retry loop; every descent landed on the puck's **right-rear edge** (full-res f00331 @ t≈11, grip 26.1 = jaws jammed on rim); puck never left the table |
| 15 | native, self-recorded (13:14) | demo-mean (nearer/left) | grasp FAIL — same peck-retry loop but misses on the **left-near edge** (full-res f00541 @ t=18); pecks dragged the puck a few cm across the table; Ctrl-C mid-retry at t≈40 |

**Scored series so far: 1/3 grasps, 0/3 full task.** Bias check verdict: **misses are NOT consistently
rightward** (right-rear in #14, left-near in #15, centered-enough in #13) → the residual +23 px camera
offset is NOT the dominant error; **±1–2 cm model scatter on the final approach is**. That points to
grasp-focused demos (data fix), not more camera calibration. Also new: #13 shows the retry behavior
re-targets the puck at a never-demonstrated position — weak but real evidence of visual servoing.
Placement note: #13's place-miss was a *release-point* error (rim, not bin center) — if picks repeat,
place-phase demos matter too. (Cosmetic: the log's "clamp 1.0 deg/tick" line prints stale MAX_STEP_DEG;
real clamps are the FAST per-motor dict — creep_test.py:189 vs :125.)

**Model identity verified for #13–15 (Mistake #1/#12 discipline):** `~/creep` (mtime 12:53, unchanged
before the 13:12–15 runs) does an **in-script** `export MODEL=…/run150k/checkpoints/last/pretrained_model`
— not a typed env prefix, so the terminal-paste failure mode doesn't apply. The same branch's
NSTEP=100/DURATION provably reached the process (log: "135s … re-plan every 100 ticks"), MODEL is
exported one line above by the identical mechanism; "Loading weights from local directory" printed
each run via `ACTPolicy.from_pretrained(MODEL)` (creep_test.py:113); the checkpoint dir holds the full
206 MB model.safetensors + processors (written 06:09 at training completion); behavior matches the
150k signature (first move 1.4–3.5° "stay at start", 4 s reach, full sequence). **Conclusion: the
scored series is a genuine read on the 150k policy.**

**Era 3 — the NSTEP=50 / retry series (2026-07-20 evening, runs #16–20):**

| # | config | puck | what happened |
|---|---|---|---|
| 16 | go50 (NSTEP=50) | left-near (off-mean) | reach+descent clean, close jammed on the rim ~45 s (grip 24–31), micro-pecking right of the puck |
| 17 | go50 | left-near (unmoved) | same rightward miss beside the puck; stopped ~44 s. Jaws opened to only 35.6 → detector arm-threshold lowered 38→34 |
| 18 | go50 | ~9 cm RIGHT of mean (overshot) | off-region **hover-freeze** 35 s+ beside the puck (the #5/#7 behavior) |
| 19 | go50 | **ON demo-mean** (ring_spot.py verified) | the true NSTEP=50 read: clean reach/descent, close ~1–2 cm LEFT, peck-jam loop. No churn at 50 — but no fix either |
| 20 | retry (NSTEP=50, RETRY=1) | on-mean | **retry never fired** — peck re-opens to 34–36 kept resetting the 10 s jam clock (threshold-coupling bug from the run-17 change). Fixed post-run: WIDE_THR=37.5 resets clocks, jam = 12 s below-wide; selftest now replays run 20's exact pattern |

**Series verdict: 0/5 grasps (cumulative 1/8, 0 full tasks).** Placement sensitivity dominated
runs 16–18 (built `eval/ring_spot.py` — live aimer, target = #13's puck spot (400,460) r55px).
Run 19 at verified demo-mean confirms: the close misses ±1–2 cm regardless of re-plan interval —
the front cam cannot see the error. **USER DECISION 2026-07-20: policy declared insufficient
as-trained; stop rollouts; run `V2_PIPELINE.md` next session. Wrist cam = one of the Anvil
cameras (physically repurposed); full demo re-record.**

**Era 4 — the v2 wrist-cam model, 150k (2026-07-22 evening):**

| # | config | puck | what happened |
|---|---|---|---|
| **21** | go2 (NSTEP=50, front+wrist, clean) | 0.8 cm right + 3.7 cm FAR of mean (user-accepted, in demo spread) | **★★★ FULL TASK, FIRST ATTEMPT.** Reach 6 s → open → descend → close t≈17 at grip **29.1** (carry band) → 8 s lift+carry → **released INSIDE the bin** (footage f00475: jaws over bin interior; f00700: puck on the bin paper, table empty) → returned to start and idled (demo episode-end behavior). The t=28 "close-on-air" line = empty jaws closing AFTER the release — detector artifact in the place phase, not a miss. Lighting slightly warmer than demos (B/R 0.76 vs 0.92) — didn't matter. Weights byte-verified in-run (150000 ckpt). User-confirmed puck in bin. |

| 28 | go2-30 (NSTEP=30) | 1.9 cm L / 4.8 FAR | NG — re-plan churn: 30 s hesitant hover-descent (Mistake #9 signature), close hit 31.3 but decayed to ~22 (slipped to rim pinch), jam verdict, gave up. N30 shelved. |
| **29** | **go2-100 (NSTEP=100, full-chunk commitment)** | 0.6 cm L / 6.5 FAR | **★★★ OK — FULL TASK.** Crisp 9 s reach, controlled descent, close t≈27 settling **32.2** (deep bite), carry at 28.7–30 with the elbow swing to the bin, release t≈45 (post-release empty close printed the usual air artifact), puck IN THE BIN (user-confirmed). Echoes v1: its breakthrough (#10/#13) was ALSO native-commitment — this policy closes best open-loop. |
| 31? | go2-100 60 (user-run, 21:56) | unrecorded | **OK (PROVISIONAL — confirm AM):** settles 36.8→25.3; end frame: table EMPTY, pucks in bin, no hand in frame. Shallow second settle keeps this uncertain. |
| 32 | go2-100 60 (user-run, 22:00) | unrecorded | NG: close-on-air 3.7 at t=18.8, no in-band grip after; end frame's empty table = the user's hand mid-reset (visible). |
| 33? | go2-100 60 (user-run, 22:02) | unrecorded | **OK (PROVISIONAL — confirm AM):** four in-band settles incl. 31.0/30.9 (deep-bite class); end frame: table empty, pucks in bin, no hand. |
| 30 | go2-10 (NSTEP=10, bracket point for the record) | 1.8 cm L / 6.8 FAR | NG — **never moved.** 135 s at the start pose, max|tgt-now| ~1.5° throughout: chunk-start ≈ current pose, so executing only 10 ticks/plan integrates to zero. The pure form of Mistake #9. |
| 22–27 | go2 (user-run; config PROVEN identical from logs: v2 weights hash line, wrist recorded, NSTEP=50) | within ~0.5–5 cm of mean (#22 nearly IDENTICAL to #21's spot); #26 8.4 cm far, #27 10 cm right | **0/6 — all grasp slips.** Signature: close CONTACTS the cup but bites shallow — grip settles 24–27 (edge clip) vs #21's 29.1 (centered wrap, matches demo carry band 26–36); cup squirts out on lift. #22 closed on air (9.2) from #21's exact spot. Eliminated by data: command/model (log-proven same), placement, object orientation (hollow-side-up in demos AND all runs — frame-compared), start pose (#22–27 started BETTER-posed than #21). Verdict: close-point scatter ~±1 cm; deep-bite rate ≈ 1/6 tonight. Wrist-view close-ups (first ever): jaws meeting the cup's curved outer wall on the shallow bites. |

Footage (all labeled OK_/NG_runNN + per-cam mp4s): `eval/run_recordings/`; run #21: `OK_run21_20260722_210042/` (grasp ~f00316, carry/release f00475, final state f00700).

**Footage evidence index** (frames in `eval/run_recordings/`): run #13 carry airborne
`run_…131209/f00376–f00439`, rim-drop `f00496–f00517`; run #14 right-rear jam `run_…131318/f00331`;
run #15 left-near miss `run_…131442/f00541`.

**Grip forensics (from parquet):** demos hold the puck at grip 26–36 while carrying (outside-wrap
around the ~7 cm puck); grip <10 while raised basically never occurs. A rollout "close" ending
below ~20 means the jaws missed the body of the puck.

**Camera bias, properly measured** (template-match on the wall outlet, conf 0.74–0.80): was
dx +48 / dy +15 px → after user nudges **dx +23 / dy +7 px (~1 cm rightward, accepted)**.
Watch for consistently rightward grasp misses — that residual bias is the prime suspect.

Key readings: the policy reaches the demos' *mean* puck zone rather than visually servo-ing to
off-mean placements (user's correct point: at 77 px offset those are nearly the same motion, so
this isn't yet proof of vision-blindness); it *does* gate on the overall scene (freeze runs 5/7);
the grasp-phase stall tracks **plan-commitment length** monotonically.

---

## 4. THE MISTAKES (read these; they cost days)

1. **THE BIG ONE — silent random weights.** `make_policy(cfg,…)` only loads weights if
   `cfg.pretrained_path` is set; programmatic `PreTrainedConfig.from_pretrained()` does NOT set
   it. No error, no warning — you get a random network with the right architecture. Every eval
   and rollout before the 07-18 fix ran garbage. **Symptoms to remember:** offline MAE ~1.0
   normalized; behavior = unnormalized mean of the dataset (the "hover at blurred average pose");
   total insensitivity to scene changes. **Fixes:** always `ACTPolicy.from_pretrained(path)`;
   when in doubt **checksum a parameter tensor against model.safetensors**; demand training-parity
   (eval loss ≈ train loss) before believing any downstream result.
2. **A whole theory built on bad data.** The "VAE latent collapse" story elegantly explained the
   random-net behavior — and was completely wrong (the latent-A/B test showed the latent
   contributes ~nothing; it's healthy). A retrain was nearly launched to fix a non-problem.
   **Lesson: before theorizing, run the decisive experiment** (here: latent on/off — which
   instead exposed the loading bug). The user's "are you actually sure or are you guessing?"
   was the single most valuable prompt of the project.
3. **Unmeasured cost claims.** Early on, "the A100 run will take a few minutes" — reality ~50 min.
   Never quote time/cost from vibes; measure a rate first (later ETAs from measured steps/s were
   accurate to minutes).
4. **Community-cloud dud GPU**: nvidia-smi fine, CUDA compute broken (err 999). **Always run a
   2-second CUDA matmul health-gate with the image's own torch BEFORE installing anything.**
5. **`pgrep`/`pkill -f` self-match** (bit us 3×): a compound command matches its *own* cmdline —
   killed its own shell twice; a completion-waiter looped **5 hours** on a 72-second eval.
   **Wait on log markers; kill via `/proc/<pid>/exe` filtering.**
6. **Buffered/filtered pipes hid errors**: `| grep | tail` swallowed the actual traceback for an
   entire debugging cycle (the "slow eval" that was actually a crash-loop). Run diagnostics
   unbuffered (`-u`) and read raw output before filtering.
7. **Unfair first offline eval**: sampled only episode 0's frozen settling-phase → false
   "two joints broken" conclusion. **Spread samples across all episodes.**
8. **The q01–q99 envelope fence**: grasp/place poses live in the top/bottom **1%** of the action
   distribution (they're the rarest frames), so a quantile "safety envelope" cut off exactly the
   behavior we wanted (48.6° of elbow!). Rails must be built from **min–max of demonstrated
   actions**, not quantiles.
9. **Creep-speed distribution shift**: demos never hover; slow, interrupted execution *creates*
   never-seen states (motionless pre-grasp poses) and re-plan churn (elbow bouncing 6→9→7→8).
   Safety rails can cause the failures they guard against. Match deployment dynamics to
   training dynamics as trust grows (clamp ≥3° needed just to beat gravity stiction; 1°/tick
   physically stalls the shoulder).
10. **Camera/lighting drift accusations — half right.** Lighting mismatch (night demos vs daylight)
    was real and fixed; camera geometry accusations went one round too far — final blend check:
    ~40–60 px ghost, effectively fine (user called it correctly). Measure (blend overlay), don't
    re-litigate. Note: phase-correlation on these scenes gives confidence 0.03 — worthless.
11. **Smaller infra stumbles**: `lerobot[training]` extra missing (caught by a free CPU smoke test
    — always smoke-test before paid runs); saved processors pin `device: cuda` (override
    `device_processor` on CPU hosts); pod rsync install ran before `apt-get update` (empty lists);
    plotted a derivative without error bars and nearly reported noise as signal (user caught it).
12. **Env-prefix commands silently didn't run** (`FAST=1 NSTEP=100 ~/creep go` — runs #8/#9
    executed at default config; two runs' conclusions were about the wrong test). **Verify config
    from the log's own printed line, never from what was typed**; bake critical configs into
    single-token launchers (this terminal mangles multi-token pastes — known machine quirk).
13. **"Watching the run" overclaim**: agent has no eyes — only telemetry — unless footage exists
    on disk. Fixed structurally: `~/creep go` now **self-records robot-view frames** every run
    (`eval/run_recordings/`), so "watch the video" is real and automatic.
14. **Margins are phase-dependent**: the ~50 px camera bias was correctly "fine" for the reach
    phase and incorrectly dismissed for the grasp phase, where ±1 cm is a third of the whole
    error budget. Re-evaluate tolerances per phase, not once globally.
15. **A busy camera looks perfectly healthy** (07-21, cost a debugging cycle). Three cameras
    `open()`ed fine, answered `VIDIOC_QUERYCAP`, enumerated MJPG+YUYV identically to the working
    one, sat in USB `runtime_status=active`, and **`fuser` reported them FREE** — yet every
    OpenCV capture failed. The holder was a `usb_cam` node inside the *privileged*
    `anvil-loader-ros2-1` container, invisible to host `fuser`/`lsof`. The only honest signal is
    **`VIDIOC_S_FMT` → `EBUSY`**, which `OPENCV_VIDEOIO_DEBUG=1` prints verbatim — that one env
    var ended the guessing. Compounding it: the docs blamed the *wrong owner entirely* (ALAN's
    8088 streamer, stopped for days) and hardcoded a stale node list. **Enumerate the actual
    holder before theorizing; a permissions/driver theory that doesn't explain "raw open works"
    is wrong.** `cam_check.py` now probes S_FMT and reads the live `/run/cameras` map.
16. **The recorder's redo/discard leaves ORPHAN FRAMES in the video files** (07-21, first v2
    session). `clear_episode_buffer()` drops the parquet rows but frames already encoded stay in
    the mp4; the next `save_episode()` then closes a video window LARGER than the episode
    (ep0: 777 rows, 1046-frame window). **Training decode is UNAFFECTED** — kept takes sit at
    the window head, orphans in the dead tail (verified visually at carry/release landmarks) —
    but `lerobot-edit-dataset` hard-asserts on the mismatch. Fix recipe: set the episode's
    `videos/*/to_timestamp` to exactly `length/fps` in `meta/episodes/…parquet`, then rerun.
    Also: in-place edits need `--new_root` EQUAL to `--root` (else output lands in
    `~/.cache/huggingface/lerobot/<repo_id>`); the tool self-backups to `<root>_old`. After a
    delete, the rewritten videos are canonical again. Expect this after every redo-heavy session.
17. **A "one-line import" can be silently dead after a repo split.** `relax.py` still imports
    `sentinel_teleop`, which stayed at `~/ALAN` — it has been unrunnable since 07-20 and nothing
    caught it because nothing ran it. `~/creep pose` is the maintained equivalent. When splitting
    a repo, grep the new tree for imports that resolve only in the old one.
18. **rsync cannot replace a directory with a symlink — and `2>/dev/null` hid it for 11 hours**
    (07-22). The pod writes `checkpoints/last` as a symlink; the local path held the v2mini
    scaffold as a real directory, so every 10-min sync failed on that one entry
    (`cannot delete non-empty directory`) while succeeding on everything else — and the DONE
    path would then have uploaded the STALE scaffold to the Hub as `ckpt-v2-last`, because its
    `model.safetensors` existence-check passed on the leftover. Two lessons: (a) never park a
    placeholder at a path an automated pipeline writes through — the "temporary" scaffold sat
    exactly where the real artifact lands; (b) a sync that "works" can still silently skip the
    one entry that matters (Mistake #6 again: the checkpoint rsync's stderr went to /dev/null).
    Verify mirrors by CONTENT (mtime/size of the file you'll ship), not by rsync's exit.

---

## 5. The learnings that now work (keep doing these)

- **The eval ladder**: offline (training-parity check!) → scene check → dry-run grounding →
  guard-railed rollout. Each rung is cheaper than the next and each caught real issues.
- **Physical grounding checks**: compare live pose vs demo-start before any run (gate >25° off);
  compare live camera vs training frames (`scene_check` page); measure puck placement vs the
  demos' region (blue-pixel detector).
- **The guardrail stack** (per-motor deg/tick clamp · min–max action envelope · start-pose gate ·
  bus watchdog · time cap · torque-hold on exit · Ctrl-C instant): 8 rollouts, zero hardware
  incidents, including the entire random-network era.
- **Cloud pattern**: API-driven deploy → CUDA health gate → pinned install → detached run with
  `.done` marker files → workcell watchdog (10-min sync, cost alert $6 / hard-kill $7.50,
  auto-terminate on completion) → all-real-numbers dashboard. The 150k run needed zero babysitting.
- **User-driven verification beats agent confidence.** The decisive corrections all came from
  the user's skepticism: the cost callout, the scene comparison demand, the puck ablation,
  "are you sure or guessing," the 77-px argument. Institutionalize: verify claims with an
  experiment before acting on them.

---

## 6. Open questions & next steps

> **➡ Active plan: `~/SO101_policy/NEXT_STEPS.md`** — 7 ranked performance options (written 07-18 after
> the #13–15 review) with costs and sequencing. Start with the free trio (ckpt A/B, NSTEP tune,
> grip-triggered retry). The items below are the older running list.

1. **[DONE — run #10]** Native commitment executed the full sequence; `~/creep go` now IS
   native mode (single token). **[DONE — runs #13–15] Scored series (partial, 3 of 5): 1/3
   grasps, 0/3 full task; first verified pick (#13).** Bias check answered: misses NOT
   consistently rightward → model scatter dominates, camera calibration is done. **User
   decision 07-18: hold off on new demos for now.** If/when resumed, the data fix is
   +15–20 grasp-focused demos *plus a few clean place-into-bin demos* (#13 missed at the
   release point), retrain ~$2.30.
2. **Localization grading** (after a successful grasp): place the puck at several in-region
   offsets; measure whether the reach target tracks it. If it doesn't and grasps only succeed
   near the mean spot → collect **+20–30 demos with deliberately varied puck positions**
   (`~/rec`), retrain (~$2.30, same pipeline) — likely the single highest-value data improvement.
3. **If the native run still stalls at the close**: suspects, in order — visual grasp-trigger needs
   exact jaw-depth match (try 2–3 cm placement/pose nudges); gripper contact/overload behavior
   (Feetech cuts torque on stall — check for overload after any close attempt); then more demos.
4. **Hygiene debt**: ~~correct the Hub model card~~ DONE 2026-07-22 (full rewrite, Hub commit
   `74bbf62`: random-net-era eval retracted with symptoms documented, all 4 checkpoint
   subfolders tabled, usage example fixed — the OLD card's own snippet reproduced the
   Mistake-#1 loading bug — and the q01–q99 envelope advice replaced with min–max; BOTH dataset cards audited 07-22 too — pick_place clarified puck-not-cube + cross-links, pick_place_v2 had NO card, full one written);
   ~~git init~~ DONE 2026-07-20; ~~ALAN's stale leader.json~~ SYNCED 2026-07-22 (was
   byte-identical to `calib_backup/leader.20260715.json`, now matches the corrected calib);
   ~~delete the /var/crash core~~ DONE 2026-07-22 (894 MB freed, no sudo needed — file was
   anvil-owned; `datasets/pick_place/images/` was found already empty);
   rigidly mount the camera (drifted twice); decide whether/when to restart the ALAN
   sentinel containers.
5. **Later**: pi0 fine-tune on the same dataset (heavier GPU, same Hub data path) once ACT is
   doing the task; Raspberry Pi deployment for the sentinel remains a separate ALAN-project goal.

---

## 7. Spend ledger

| item | cost |
|---|---|
| 20k proof run (incl. dud 4090 + A100) | ~$1.46 |
| 150k full run (A5000, 8 h) | $2.21 |
| v2 150k run (A4500, 21 h, 2-cam) | $5.40 |
| **total** | **~$9.07** — RunPod balance ≈ $0.92 of $10 |

*Doc maintained by Claude; update it at every major result. Last update 2026-07-23 ~09:30 (loss-chart suite added; §0 07-23 bullet). MORNING PICKUP LIST: (1) user confirms/corrects the PROVISIONAL
OK labels on runs #31/#33 (relabel with `python eval/score_run.py` + fix this table if wrong);
(2) clean N100 series — `~/creep go2-100 135`, puck anywhere in-region, answer o/n at the prompt
(everything auto-labels + encodes to eval/run_recordings/); (3) gripper overload/torque check
before the series (repeated closes may trip Feetech protection — if real, tonight's rates
UNDERSELL the policy); (4) the retrain decision: grasp-focused demos + retrain ≈ $4–5, balance
$0.92 → user top-up call; chunk_size=120 (longer commitment) is a candidate config for that same
run. NOTE: `~/creep` gained go2-10/30/100 tonight and is now backed up at `tools/creep.sh` —
sync the two after edits. All 13 runs of 07-22 are labeled OK_/NG_runNN in eval/run_recordings/
with front+wrist mp4s (runs #21–29 chronology: #31–33 actually ran BETWEEN #29 and #30, numbers
assigned at labeling time).*