# Policy Training v3: SO101 pick-and-place comparative ACT run

Last updated: 2026-08-06

This document is the full record of the `pick_place_v3` experiment: why it
exists, what was done from first smoke demo through training and retention, what
went wrong, what is still running, and how to finish and evaluate it.

It is meant to be readable top-to-bottom. Later sections are reference depth
for the same story, not a second competing timeline.

This document intentionally does not contain API tokens, private keys, or
private-key contents. RunPod public IPs and forwarded SSH ports are ephemeral;
confirm them in the RunPod console before using any host/port example.

---

## How to read this document

| If you need… | Start here |
|---|---|
| The point of the experiment | §1 |
| What happened from start to now | §2 |
| What is done vs still running | §3 |
| How to roll out the finished stock policy | §4 |
| How each pipeline stage works | §5–§11 |
| Exact HF repos, metrics, SHAs | §12 |
| Mistakes not to repeat | §13 |
| How to run the next experiment cleanly | §14 |

---

## 1. Purpose of the experiment

The question is whether LMFAO augmentation improves physical SO101
pick-and-place robustness under matched ACT training.

It is **not**:

- a loss-only benchmark
- a reuse of the old `pick_place_v2` smoke dataset as the production corpus
- a comparison of different ACT architectures or hyperparameter sweeps

The controlled design is:

1. Collect one fresh 40-episode dual-camera stock dataset (`pick_place_v3`).
2. Build five LMFAO augmentation buckets from that same stock.
3. Train six ACT policies with identical hyperparameters and seed.
4. Change only the training dataset between policies.
5. Compare physical rollout success in nominal and held-out conditions.

The six training buckets:

| Bucket | Contents |
|---|---|
| `stock` | 40 untouched demos (26,021 frames) |
| `lighting` | originals + brightness / contrast / color-temperature variants |
| `noise` | originals + Gaussian / uniform sensor-noise variants |
| `occlusion` | originals + border / sequence-box / moving-box variants |
| `spatial` | originals + random-crop variants |
| `full` | originals + combined lighting + Gaussian noise + border occlusion + crop |

Each augmented bucket has:

- 40 source episodes
- 8 variants per source
- originals retained with `--include-original`
- 360 total episodes
- 234,189 frames
- front + wrist cameras
- 720 MP4 files after LMFAO export

Within an augmented bucket, episodes are grouped by source: original first,
then variants 0–7. Originals are ~11% of the bucket; augmented examples ~89%.

Task string used throughout:

```text
pick the blue puck and place it in the brown box
```

---

## 2. End-to-end story of this run

This section is the narrative spine. Everything later is detail for these
phases.

### 2.1 Setup and smoke

Work began by proving the path on the NUC before collecting production data:

- Confirm arms, dual cameras, calibration, and LeRobot 0.6 on
  `multiply@100.103.79.98`.
- Record a tiny dual-camera smoke set.
- Augment it, write LeRobot v3, and run a short GPU ACT smoke.

Smoke findings that shaped production:

- Early augmentation could silently emit front-only output from dual-camera
  input. The CLI was hardened and multi-camera replay was fixed.
- LeRobot v3 writers needed exact schema (`stats`, `tasks.parquet`, typed
  state/action, full `info.json` columns).
- ACT could train on the fixed dual-camera augmented smoke (loss fell roughly
  74 → 22 over ten steps).

### 2.2 Fresh collection

Forty production demos were recorded on the NUC into:

```text
/home/multiply/LMFAO/so101/datasets/pick_place_v3
```

Both cameras stayed enabled. Scene, lighting, and mounts were held fixed.
Faulted episodes were discarded rather than “corrected” into the corpus.

Stock video layout note: native LeRobot sharding packs the 40 stock episodes
into two shared MP4 shards per camera. An early check incorrectly expected one
file per episode; the smaller stock file count was normal, not data loss.

### 2.3 Augmentation on the NUC

Production buckets were built under:

```text
/home/multiply/LMFAO/eval_buckets_v3
```

via `scripts/augment_eval_buckets.sh` with `--variants 8 --include-original
--seed 7 --resume`.

Production fixes discovered here:

- **Memory:** materializing eight dual-camera variants at once OOMed the NUC.
  Augmentation was changed to stream one produced episode at a time.
- **Replay:** Gaussian vs uniform noise used different parameter names
  (`sigma` vs `amplitude`); secondary-camera replay was corrected.
- **Schema:** writers were brought into strict LeRobot v3 compliance.

Buckets were validated for episode/frame counts, both cameras, decode, and
metadata before any cloud training.

### 2.4 Transfer onto RunPod

Six CUDA pods were provisioned, one per bucket:

```text
lmfao-act-stock / lighting / noise / occlusion / spatial / full
```

Data moved with resumable `rsync` (and some pod-to-pod copies for large
buckets). The Mac was deliberately not used as a multi-GB staging disk.

Transfer was painful and needed hardening, but it was **not** the root cause of
later slow training.

### 2.5 Training launch and the GOP crisis

All six policies used the same ACT recipe: 100k steps, batch 8, seed 7,
4 workers, prefetch 4, PyAV, checkpoints every 20k, no Hub push during train.

Stock trained at a healthy ~6.5 steps/s. Augmented buckets initially crawled
(~0.4–3 steps/s). Loader worker tuning and TorchCodec experiments did not fix
it.

Root cause: LMFAO-exported H.264 videos had ~250-frame GOPs. ACT samples frames
randomly; PyAV had to seek to a distant keyframe and decode forward, starving
the GPU.

Fix: re-encode augmented videos on each pod with GOP=2 (`scripts/reencode_low_gop.py`).
Healthy pods then recovered to ~6.4–6.6 steps/s.

Stock was never the GOP victim in the same way; its native recorder shards
behaved acceptably.

### 2.6 Operational scars during training

Several self-inflicted costs happened mid-flight:

- Renaming live pods restarted containers, broke `/root`-based Python
  symlinks, forced re-encodes to resume, and cost stock ~8.7k uncheckpointed
  steps (resume from 80k after reaching ~88.7k).
- Lighting/noise stayed slow (~1–2 steps/s) even after low-GOP re-encode on a
  shared slow host/storage path. Mid-run kill/restart “fixes” were rejected;
  those jobs were left running.
- Full-pod env rebuilds after rename were expensive; the durable pattern became
  cloning a proven Python/CUDA tree between pods, not reinstalling from scratch
  on every machine.

Standing rule from this phase: **never mutate a pod that is still training**.

### 2.7 Retention of finished work

When a bucket reached 100k:

1. Validate `checkpoints/last/pretrained_model`.
2. Upload deployable policy root + `training_summary.json` to private HF.
3. Upload matching LeRobot datasets.
4. Verify remote file counts / bytes / model SHA-256.
5. Then stop the idle pod (after operator approval / explicit cleanup step).

Spatial’s dataset upload was interrupted at 502/1085 files and later resumed to
a byte-identical 1085-file completion. Occlusion policy + dataset uploaded
cleanly in one pass (1085 data files, byte-identical).

Stock, spatial, and occlusion pods were stopped after HF verification. Full
finished the same day and followed the same path.

Lighting and noise never recovered from their slow host (~2 steps/s with ~11–14
h remaining). With the RunPod budget nearly exhausted, the operator cancelled
both on 2026-08-06: lighting's 20k checkpoint was saved to HF, noise had no
checkpoint and was lost, and all eight pods (six originals + two aborted
migration pods) were deleted. The RunPod phase is closed; no billing remains.

### 2.8 Physical evaluation (in progress / operator-owned)

NUC rollouts are handled separately by the operator. The durable policy sources
are the private Hugging Face repos, not RunPod. The comparison set is now four
complete policies (stock, spatial, occlusion, full); lighting exists only as a
partial 20k checkpoint and noise was lost.

---

## 3. Final status (run ended 2026-08-06)

The RunPod phase is **closed**. All pods were deleted on 2026-08-06 ~20:24
America/New_York to stop billing (budget exhausted). No RunPod resources
remain.

| Bucket | Training | Final artifact | Rollout (nominal) |
|---|---|---|---|
| stock | 100k complete | policy + dataset verified on HF | **11/15** |
| spatial | 100k complete | policy + dataset verified on HF | not yet run |
| occlusion | 100k complete | policy + dataset verified on HF | **10/15** |
| full | 100k complete | policy + dataset verified on HF | **10/15** |
| lighting | cancelled at ~21k / 100k | partial 20k checkpoint saved on HF | n/a |
| noise | cancelled at ~15k / 100k | **lost** — no checkpoint ever written | n/a |

Lighting and noise never exceeded ~2 steps/s because both pods landed on the
same slow-host storage path (measured ~1 item/s random access vs ~6.5 on
healthy hosts). Low-GOP re-encode, worker tuning, and local-overlay copies did
not fix it. A migration to two fresh 4090 pods (`lmfao-act-lighting-r2`,
`lmfao-act-noise-r2`) was provisioned and keyed, but the operator cancelled
before the transfer started and all pods were deleted.

Noise's ~15k steps were lost because the first checkpoint (20k) had not yet
been written. Lighting's 20k checkpoint was saved to
`Dillonjohnson/pick_place_v3_act_lighting` (marked partial, not comparable to
the 100k policies).

If lighting/noise are ever retrained: provision on a healthy host, benchmark
random-access decode **before** launch, transfer the NUC-held buckets (originals
still exist at `/home/multiply/LMFAO/eval_buckets_v3`), re-encode low-GOP, and
train from scratch.

---

## 4. What to do next: roll out stock on the NUC

Stock is finished and retained. Prefer Hugging Face over the stopped pod.

```text
Policy:  Dillonjohnson/pick_place_v3_act_stock
Dataset: Dillonjohnson/pick_place_v3_stock
```

### 4.1 Pull this runbook onto the NUC

Commit/push from the Mac first, then:

```bash
ssh multiply@100.103.79.98
cd /home/multiply/LMFAO
git pull
less docs/policytraining_v3.md
```

### 4.2 Get the policy onto the NUC

```bash
cd /home/multiply/LMFAO
mkdir -p policies
source /home/multiply/envs/lerobot06/bin/activate
huggingface-cli download Dillonjohnson/pick_place_v3_act_stock \
  --local-dir /home/multiply/LMFAO/policies/stock

test -f policies/stock/model.safetensors
test -f policies/stock/config.json
test -f policies/stock/policy_preprocessor.json
du -sh policies/stock   # expect ~200 MB
```

Or pass `--policy.path=Dillonjohnson/pick_place_v3_act_stock` directly after
Hub login.

Do **not** copy only `model.safetensors`. LeRobot needs the full deployable
root (config, processors, train config).

Historical note: before stop, NUC key `~/.ssh/lmfao_runpod_transfer` could
`scp` from stock at `209.170.80.132:14132`. That endpoint is gone while the
pod is EXITED. Prefer HF. If the pod is later resumed, reconfirm IP/port and
re-authorize the NUC public key.

### 4.3 Verify robot and cameras

The stock policy expects:

- `observation.state`: 6D
- `observation.images.front`: RGB 640×480
- `observation.images.wrist`: RGB 1280×720
- `action`: 6D
- 30 FPS

Pins change after USB replugs. Verify:

```bash
cd /home/multiply/LMFAO
source so101/env.sh
python so101/eval/cam_check.py
python so101/eval/live_view.py
python so101/eval/read_pose.py 3
readlink -f /dev/so101_follower
cat so101/front_cam.path so101/wrist_cam.path
```

Do not use `WRIST=0`, swap camera names, or feed both names from one UVC
device. Snapshot pins during this run were often `front=/dev/video6`,
`wrist=/dev/video0` — treat as historical, not gospel.

### 4.4 Five-second safety smoke, then longer

Safety first: clear the swept volume, keep e-stop ready, start in the nominal
training layout, do not connect the leader for base rollout, expect follower
torque on connect.

```bash
cd /home/multiply/LMFAO
source so101/env.sh

FRONT="$(readlink -f "$(cat so101/front_cam.path)")"
WRIST="$(readlink -f "$(cat so101/wrist_cam.path)")"

CAMERAS="{ \
front: {type: opencv, index_or_path: ${FRONT}, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
wrist: {type: opencv, index_or_path: ${WRIST}, width: 1280, height: 720, fps: 30, fourcc: MJPG} \
}"

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=follower \
  --robot.cameras="$CAMERAS" \
  --policy.path=/home/multiply/LMFAO/policies/stock \
  --policy.device=cpu \
  --device=cpu \
  --task="pick the blue puck and place it in the brown box" \
  --fps=30 \
  --duration=5 \
  --display_data=true \
  --return_to_initial_position=true
```

Use `lerobot-rollout`, not `lerobot-record`, for policy deployment on this
LeRobot version. The NUC has no CUDA GPU — keep both device flags on `cpu`.

If the 5 s smoke is safe, repeat at `--duration=20`. Manually log policy name,
time, scene, success/failure, failure mode, and whether a human stopped it.

---

## 5. Machines and responsibilities

Three machine classes. Mixing their roles caused most operational pain.

### Development Mac

- Edit code, tests, docs
- Provision / monitor / stop RunPods
- Orchestrate HF uploads (token stays on Mac; passed in-memory to pods)
- **Not** a staging disk for multi-GB datasets or CUDA envs
- **Never** store RunPod private keys in the repo

### SO101 NUC (`multiply@100.103.79.98`)

- Teleop and recording
- CPU augmentation
- Episode review
- Physical rollout

Important paths:

```text
/home/multiply/LMFAO
/home/multiply/envs/lerobot06
/home/multiply/LMFAO/so101/datasets/pick_place_v3
/home/multiply/LMFAO/eval_buckets_v3
```

### RunPod CUDA pods

- GPU ACT training
- Temporary datasets / checkpoints
- Low-GOP re-encode of augmented videos

Canonical map (all pods deleted 2026-08-06; IDs kept for historical reference):

```text
lmfao-act-stock      fnre6t63nr42wq   DELETED after HF retention
lmfao-act-spatial    q7k4rlc4d5cp0w   DELETED after HF retention
lmfao-act-occlusion  qmpthutzu6r2dl   DELETED after HF retention
lmfao-act-full       q5vlgbs5tnxsyd   DELETED after HF retention
lmfao-act-lighting   7k25sdlk1rkgi1   DELETED (training cancelled ~21k)
lmfao-act-noise      p3zptith51vy9b   DELETED (training cancelled ~15k)
lmfao-act-lighting-r2 mo9bybj9wr1pdl  DELETED (migration aborted)
lmfao-act-noise-r2   xjctrsghb2ezbf   DELETED (migration aborted)
```

Name pods before work starts. Stopping, renaming, or resizing a live trainer
is forbidden unless the operator explicitly approves it.

---

## 6. Phase detail: data collection

Production corpus is `pick_place_v3`, not Hub `Dillonjohnson/pick_place_v2`.

```bash
cd /home/multiply/LMFAO
./scripts/record_demos.sh 40
# or: ./so101/rec 40
```

Protocol:

1. Fix lighting, mounts, and table scene.
2. Place puck in the nominal start area.
3. Match leader/follower poses before connecting.
4. Teleoperate slowly and consistently.
5. ENTER to stop each episode.
6. Keep good episodes only; redo faults; do not save correction episodes.
7. Keep both cameras enabled.

The guided recorder rejects empty/faulted episodes and can resume a dataset
only when camera keys match.

Review samples copied for manual viewing during the run (not training inputs):

```text
~/Downloads/lmfao_stock_samples/stock_front.mp4
~/Downloads/lmfao_stock_samples/stock_wrist_top.mp4
```

---

## 7. Phase detail: augmentation

### 7.1 Production command

```bash
cd /home/multiply/LMFAO
export STOCK=/home/multiply/LMFAO/so101/datasets/pick_place_v3
export OUT_ROOT=/home/multiply/LMFAO/eval_buckets_v3
export VARIANTS=8
export SEED=7
./scripts/augment_eval_buckets.sh
```

Creates `stock` as a symlink to the source and runs `lmfao augment` with
`--variants 8 --include-original --seed 7 --resume`.

Configs live under `configs/training/buckets/`.

### 7.2 Exact bucket transforms

Lighting:

```json
[
  {"name": "lighting.brightness", "params": {}, "probability": 1.0},
  {"name": "lighting.contrast", "params": {}, "probability": 1.0},
  {"name": "lighting.color_temperature", "params": {}, "probability": 1.0}
]
```

Noise:

```json
[
  {"name": "noise.gaussian", "params": {"sigma": 0.03}, "probability": 1.0},
  {"name": "noise.uniform", "params": {}, "probability": 1.0}
]
```

Occlusion:

```json
[
  {"name": "occlusion.border_intrusion", "params": {}, "probability": 0.7},
  {"name": "occlusion.sequence_box", "params": {}, "probability": 0.5},
  {"name": "occlusion.moving_box", "params": {}, "probability": 0.5}
]
```

Spatial:

```json
[
  {"name": "spatial.random_crop", "params": {}, "probability": 1.0}
]
```

Full (sequential combined pipeline, not disjoint families):

```json
[
  {"name": "lighting.brightness", "params": {}, "probability": 1.0},
  {"name": "lighting.contrast", "params": {}, "probability": 1.0},
  {"name": "lighting.color_temperature", "params": {}, "probability": 1.0},
  {"name": "noise.gaussian", "params": {"sigma": 0.03}, "probability": 1.0},
  {"name": "occlusion.border_intrusion", "params": {}, "probability": 0.5},
  {"name": "spatial.random_crop", "params": {}, "probability": 1.0}
]
```

A `full` variant receives brightness, contrast, color temperature, Gaussian
noise, and random crop in the same generated video; border occlusion is also
applied with probability 0.5.

### 7.3 Production fixes that had to land first

**Bounded memory.** One dual-camera episode uncompressed is multi-GiB; eight
variants at once killed the first lighting job. Production now streams via
`iter_augment_episode` / `iter_sweep_episode` in `src/lmfao/program.py` and
`src/lmfao/cli.py`.

**Dual-camera replay.** Transform history is replayed onto the wrist stream.
Gaussian (`sigma`) and uniform (`amplitude`) are handled separately in
`src/lmfao/replay.py`.

**LeRobot v3 schema.** Writers emit `meta/stats.json`, correct
`tasks.parquet`, full `info.json` column declarations, and
`fixed_size_list<float32>[dim]` for state/action.

**Train output dirs.** LeRobot requires a fresh `output_dir` when
`resume=false`. Scripts create only the parent and let `lerobot-train` create
the run directory (`scripts/train_act.sh`, `scripts/train_eval_buckets.sh`).

### 7.4 Pre-cloud validation checklist

For each augmented bucket:

- complete LeRobot metadata
- 360 episodes / 234,189 frames
- both camera keys present and decodable
- state/action dims correct
- no macOS AppleDouble `._*` files
- CUDA available on the target pod

---

## 8. Phase detail: transfer to RunPod

Evolution of the transfer path:

1. Direct tar streams — brittle
2. File-by-file Python resume — workable but slow to operate
3. `rsync --partial --append-verify` — robust NUC→pod method
4. Pod-to-pod copies for `full` / `occlusion` — avoid Mac hairpin

Robust pattern:

```bash
rsync -rt \
  --no-owner --no-group --no-perms \
  --partial --append-verify \
  --info=progress2 \
  -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=12 \
      -i KEY -p PORT" \
  SOURCE/ root@HOST:DEST/
```

Transfer lessons:

- Always resume; always verify file count and bytes after.
- Do not route large buckets through the Mac.
- If a Mac-originated tar is unavoidable: `COPYFILE_DISABLE=1`.
- Pod SSH endpoints change after restart; names/IDs are the durable handles.
- A pod rename can restart the container and change the forwarded port.

---

## 9. Phase detail: ACT training

### 9.1 Matched configuration

```text
policy:              ACT (~52M params)
steps:               100000
batch size:          8
num_workers:         4
prefetch_factor:     4
seed:                7
device:              CUDA
push_to_hub:         false
save_checkpoint:     true
save_freq:           20000
log_freq:            200
wandb:               disabled
video_backend:       pyav
```

Representative launch:

```bash
lerobot-train \
  --dataset.repo_id=local/lmfao_eval_BUCKET \
  --dataset.root=/workspace/eval_buckets/BUCKET \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=8 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --seed=7 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --log_freq=200 \
  --wandb.enable=false \
  --output_dir=/workspace/runs/eval_buckets/BUCKET
```

Deployable checkpoint size is ~200 MB per policy.

### 9.2 Known-working RunPod environment

```text
Python:      3.12.13
LeRobot:     0.6.2 @ ef88d4e52b9f3a16638e4b73202d619b7606fd41
Torch:       2.11.0+cu128
Torchvision: 0.26.0+cu128
PyAV:        15.1.0
TorchCodec:  0.15.0
Accelerate:  1.14.0
Datasets:    5.0.1
```

LeRobot at this revision needs Python ≥ 3.12 and Torch ≥ 2.7. A base image
with Python 3.11 / Torch 2.4 is not a valid substitute. Keep PyAV on the 15.x
line.

After rename breakage, the recovery pattern was: copy a proven `/workspace`
Python runtime (~110 MB) between pods and recreate expected symlinks — never
stage that tree on the Mac.

### 9.3 The long-GOP bottleneck (root cause)

Augmented MP4s had ~250-frame keyframe gaps. ACT random access forced long
decode tails.

Random-access microbench on one production clip:

```text
original long-GOP:   0.33 independent frame requests/s
low-GOP (GOP=2):     2.76 independent frame requests/s
improvement:         ~8.4×
```

Re-encode settings used on pod copies (NUC originals untouched):

```text
codec: libx264
preset: veryfast
CRF: 18
GOP / min keyframe interval: 2
scene-cut keyframes: disabled
pixel format: yuv420p
tool: scripts/reencode_low_gop.py
```

The tool skips already-fixed files, writes `*.low-gop.tmp.mp4`, validates
frame/packet counts and keyframe spacing, atomically replaces, and excludes
its own temps from discovery.

Post-fix healthy throughput:

```text
stock:      ~6.5–6.6 steps/s
spatial:    ~6.4 steps/s
occlusion:  ~6.3–6.6 steps/s
full:       ~6.5 steps/s
```

Lighting and noise remained ~1–2 steps/s after the same re-encode on a slower
host/storage path. That is a remaining systems issue, not a reason to kill the
jobs mid-run.

NVENC was advertised but failed on the 4090 (`unsupported device`); CPU
`libx264` was used after a validated single-file test. Always pass
`ffmpeg -nostdin` inside SSH heredocs.

---

## 10. Phase detail: Hugging Face retention

### 10.1 Layout

Hugging Face has no nested `owner/v3/stock` path. Use one private collection
plus separate repos, each usable directly as `--policy.path`.

```text
Collection:
https://huggingface.co/collections/Dillonjohnson/so101-pick-place-v3-act-policies-6a74e56a093a15c8acc55786

Policies:
Dillonjohnson/pick_place_v3_act_stock      complete
Dillonjohnson/pick_place_v3_act_spatial    complete
Dillonjohnson/pick_place_v3_act_occlusion  complete
Dillonjohnson/pick_place_v3_act_full       complete
Dillonjohnson/pick_place_v3_act_lighting   partial (20k/100k checkpoint)
Dillonjohnson/pick_place_v3_act_noise      empty (training cancelled pre-checkpoint)

Datasets:
Dillonjohnson/pick_place_v3_stock          complete
Dillonjohnson/pick_place_v3_spatial        complete
Dillonjohnson/pick_place_v3_occlusion      complete
Dillonjohnson/pick_place_v3_full           complete
```

Deployable files live at the **model repo root**. Do not upload credentials,
private keys, or intermediate incomplete checkpoints. Pass the HF token to
pods only in memory.

### 10.2 Extraction checklist (every finished bucket)

1. Confirm `.done` and complete `checkpoints/last/pretrained_model`.
2. Upload the full deployable folder to the matching private model repo.
3. Write `training_summary.json` (metrics, model bytes, SHA-256).
4. Upload the matching LeRobot dataset when retaining that bucket.
5. Verify remote file counts and byte sizes against the pod.
6. Only then: `runpodctl pod stop <id>`.

### 10.3 Shutdown history (run closed)

```bash
export RUNPOD_API_KEY="$(<"$HOME/.config/lmfao/runpod_api_key")"
# stopped after HF verification:
runpodctl pod stop fnre6t63nr42wq   # stock
runpodctl pod stop q7k4rlc4d5cp0w   # spatial
runpodctl pod stop qmpthutzu6r2dl   # occlusion
runpodctl pod stop q5vlgbs5tnxsyd   # full

# final teardown 2026-08-06 ~20:24 ET (budget exhausted) — deleted ALL pods,
# including the two aborted migration pods:
runpodctl pod delete 7k25sdlk1rkgi1   # lighting (training cancelled)
runpodctl pod delete p3zptith51vy9b   # noise (training cancelled)
runpodctl pod delete mo9bybj9wr1pdl   # lighting-r2 (aborted)
runpodctl pod delete xjctrsghb2ezbf   # noise-r2 (aborted)
runpodctl pod delete fnre6t63nr42wq q7k4rlc4d5cp0w qmpthutzu6r2dl q5vlgbs5tnxsyd
```

`runpodctl pod list` returned `[]` after teardown. No RunPod billing remains.

---

## 11. Phase detail: physical comparison plan

Training loss is not the experiment result. Use physical success rate.

### 11.0 Results so far (2026-08-06, nominal layout)

| policy | placed | failed at the grasp | failed after grasping | run |
|---|---|---|---|---|
| stock | **11/15 (73%)** | 4 | 0 | `2026-08-06_1833_stock` |
| occlusion | **10/15 (67%)** | 5 | 0 | `2026-08-06_1928_occlusion` |
| full | **10/15 (67%)** | 5 | 0 | `2026-08-06_2034_full` |

(Failure-mode columns corrected 2026-08-09 after frame-by-frame re-audit of
all 14 failure clips — see the paragraph below. The placed/failed counts are
unchanged.)

All three land within one trial of each other, which at n=15 is
indistinguishable. **No augmentation family beat stock in the nominal layout,
and none collapsed either.**

The second half of that is what matters here. The criterion in §11.2 is that a
family is useful when it beats stock in its *matching held-out* condition
without collapsing nominal success. Nominal is intact for all three, so nothing
has been ruled out — these runs are the control, and the held-out conditions
are still to run.

Worth noting what does *not* predict physical success: final loss and offline
MAE both order stock -> spatial -> occlusion -> full, exactly by augmentation
aggressiveness (0.053 / 0.061 / 0.064 / 0.068; 0.86 / - / 1.02 / 1.07 deg).
None of that ordering appears in the rollout results. That is the whole reason
the experiment is scored on the robot. Both runs are 60 s per trial with 25 s resets, identical gates, and the
same policy configuration; the only difference is the training corpus.
Per-episode clips and a synced viewer are under each run's `clips/` and
`web/index.html` (`so101/eval/make_viewer.py`).

Where they fail is more informative than the rate, and the failure modes were
re-audited frame-by-frame on 2026-08-09 (all 14 scored failure clips, tracking
the puck in the front view; wrist confirmation where ambiguous). **Every
scored failure of every policy is the same failure: the missed close** — the
grasp missing by roughly a centimetre, the puck nudged rather than secured,
and the policy running the rest of the script with empty jaws (see the note
below). The puck never leaves the table in any of the 14. An earlier version
of this paragraph claimed occlusion twice dropped the puck on the rim and full
once released it mid-air away from the box; those were front-camera misreads
of the empty-jaw dwell at the rim, and with them goes the claim that the
augmented policies "reach further into the task." There is no evidence the
families fail differently. Two review-harness caveats this exposed: "held in
band" time registers for empty jaws transiting the carry band, and "puck
moved" can reflect a nudge during the missed close — neither implies the puck
was ever grasped.

One more failure-behavior note, from frame-by-frame review (front + wrist) of
the unscored `2026-08-06_2023_stock` run (3 episodes, all missed closes):
after a miss, the policy does not re-attempt — it runs the rest of the script
with empty jaws. It lifts, transports to the box, and dwells ~10 s pressing
nothing against the rim, because there is no grasp-verification signal to tell
it the puck was never secured; the puck stays on the table, knocked beside the
box by the closing jaws. Two practical consequences. First, scoring this
correctly requires tracking the *puck*, not the arm: in the front view the
puck remains visible on the table for the entire trial, so a careful
frame-by-frame watch is by itself sufficient to call the missed close (the
wrist view confirms it directly — jaws closed on empty air). A reviewer who
follows the arm instead will misread the empty-jaw dwell at the rim as a place
failure. Second, the missed close is the single point of leverage: the one
recovery on record (ep9 in §11.0.1) happened because a second approach attempt
occurred, so a grasp-check / re-approach behavior addresses the dominant
failure of every policy tested.

Occlusion also behaves differently *after* succeeding: it keeps moving and
grabs at empty air with the puck plainly sitting in the box, markedly more than
stock. Neither policy has a termination signal so some of this is expected, but
the degree differs. Worth testing with a vision ablation (hold
`observation.state` fixed, swap only the camera images, measure how far the
predicted action moves) — occlusion augmentation trains a policy to keep
working when parts of the image are blocked, which could plausibly teach it to
depend less on vision overall. Stock's numbers for that comparison: images from
another episode move the action 8.5-15.1 deg, black frames 42.5 deg, random
noise 57.6 deg. A vision-blind policy returns ~0.

### 11.0.1 Stock baseline detail

**11 / 15 = 73%** in the nominal layout, 60 s per trial, recorded at
`so101/eval/run_recordings/2026-08-06_1833_stock`.

```text
ep  0  FAIL      ep  5  PASS      ep 10  PASS
ep  1  FAIL      ep  6  PASS      ep 11  PASS
ep  2  PASS      ep  7  PASS      ep 12  PASS
ep  3  FAIL      ep  8  PASS      ep 13  FAIL
ep  4  PASS      ep  9  PASS      ep 14  PASS
```

For scale: v1 (front-cam only) never placed the puck in 8 lifetime rollouts,
and v2's best was ~3/6. See `so101/CHECKPOINT.md`.

The failure mode is unchanged from v1/v2 and it is entirely the close. Time
spent holding the puck inside the demos' carry band separates the two outcomes
without overlap:

```text
failures:   0.7, 1.3, 1.3, 2.7 s
successes:  4.6 - 5.4 s
```

So the reach and the transport are reliable; grasping is what scatters. When it
secures the puck it finishes the task nearly every time.

Two observations worth carrying into the augmented buckets:

- Three of the four failures were in the first four trials, after which it went
  10/11. If that is a warm-up effect rather than variance, it biases whichever
  policy is run first — so run the same warm-up for every bucket.
- One success (ep9) recovered: the first approach missed, it backed off,
  re-approached and placed. v1/v2 never recovered from a missed close; they
  pecked until the clock ran out. Clip in `clips/ep9_recovery_*.mp4`.

### 11.1 Running a bucket

```bash
./so101/series 15                 # stock
POLICY=spatial ./so101/series 15  # any other bucket in policies/
```

Each trial homes the arm to the demos' mean start pose, gates the setup, then
records. `./so101/eval_py review.py <run-dir>` scores it.

Everything the gate compares against is derived from `pick_place_v3` at run
time — camera geometry from the dataset's feature shapes, start pose and puck
placement from its episodes, the target box from where the puck ends up in the
demos. Nothing is written into the scripts, because a hardcoded constant is how
`ring_spot.py` came to aim at pixel (756,584) on a 640x480 frame.

Three harness bugs this caught, all of which would have been scored as policy
failures:

- **Camera warm-up.** LeRobot defaults `warmup_s=1`; the RealSense front camera
  needs ~3 s for auto-exposure/white-balance to converge (B/R 0.518 -> 0.892,
  brightness 38 -> 112, against demos at 0.910 +/- 0.012 and 113.0 +/- 1.5). At
  the default, the policy's first observation is ~20 sigma off-distribution and
  ACT commits a whole 100-action chunk from it. Now `warmup_s=4`.
- **Start-pose sag.** The follower is left limp on disconnect and sags under
  gravity — 5.4 deg at `shoulder_lift` between two rollouts, which is several cm
  at the gripper against a ~1 cm grasp budget. `so101/eval/goto_start.py` drives
  it back and holds torque.
- **Scoring on the wrong signal.** Peak grip marks the *approach* (the jaws open
  to ~39 before closing to ~29.6 to carry), so scoring on it calls the human
  demos themselves failures. Score on whether the puck reached the box.

### 11.2 Conditions still to run

Suggested conditions (same trial count per policy/condition; 20 is a reasonable
first target):

- nominal training layout
- dimmer lighting
- warmer lighting
- slight front-camera framing shift
- slight wrist-camera framing shift
- small edge occluder
- exposure/gain disturbance

Per trial record:

- policy bucket + checkpoint step
- condition
- success/failure
- grasp / transport / placement success
- collision or unsafe motion
- human intervention
- short failure note

An augmentation family is useful when it beats stock in its matching held-out
condition without collapsing nominal success.

### 11.3 Diagnosis: why augmentation didn't move the nominal number (2026-08-09)

Full frame-by-frame audit of all 14 scored failure clips (front view, tracking
the puck; wrist confirmation where ambiguous) plus a joint-pose analysis of
every demo and rollout episode (`so101/eval/diagnose_close.py`). Four pillars:

1. **The nominal layout was the control arm, and it returned the expected
   null.** Augmentation buys robustness under distribution shift; nominal has
   no shift, so parity (73/67/67, indistinguishable at n=15) is what theory
   predicts. The runs that test the actual hypothesis are the §11.2
   conditions, none of which have run yet.
2. **The dominant failure is outside the augmentation's action space.** All
   17 verified failures (14 scored + 3 unscored stock) are the same ~1 cm
   missed close followed by an empty-jaw place script. LMFAO's ADJUST path is
   pixel-only — `observation.state` and `action` are copied unchanged — so it
   trains *invariance* ("same action despite pixel changes"), while the
   bottleneck demands *sensitivity* to sub-cm puck displacement. Two v1
   ingredients plausibly hurt precision: per-frame ±8 px crop jitter (≈ the
   miss scale on the front camera, paired with unchanged labels) and
   occluders that can cover the wrist-view close.
3. **The close is dead-reckoned, and its scatter is the size of the
   tolerance.** ACT ran `chunk_size=100`, `n_action_steps=100`,
   `n_obs_steps=1`, no temporal ensembling: the policy commits 3.3 s of
   actions from one observation, so the sub-second close executes open-loop.
   Rollout close poses scatter 1.5–6° per joint (≈1–3 cm at the gripper)
   against a ~1 cm capture tolerance — success is partly placement luck.
   Close-pose offsets vs the demo mean are all within ~1.3σ, so it is not a
   gross calibration offset. Stock/full failures are directionally random;
   occlusion's five failures share one systematic offset (shoulder_lift
   +8.3°, elbow −9.7°, t≈5) and the earliest close time (3.7 s vs stock
   4.3 s, demos 4.6 s) — the signature of closing blind, consistent with
   occluders masking the close in training. Small n; suggestive, not
   settled.
4. **Half the protocol never reached the robot, and the instruments
   misled.** Spatial OOMed twice, lighting was killed at epoch 14, noise
   OOMed at epoch 4 — three of six planned policies never ran. And the
   review harness's proxy metrics ("held in band", "puck moved") produced
   the phantom rim-drop / mid-air-release failure modes corrected in §11.0:
   band time registers for empty-jaw transits and puck-moved registers
   nudges, so neither implies a grasp.

Also measured: demo close poses span 63–75° in shoulder_lift/elbow across
the 40 demos (puck placed all over the workspace), so local density at the
nominal eval spot is sparse; and the augmented buckets' higher final losses
(0.061–0.068 vs stock 0.053) track repetition, not quality — 100k steps is
~31 passes over stock's 26k frames but only ~3.4 passes over each bucket's
234k frames.

Consequences, ranked by leverage: (1) grasp verification + re-approach —
every failure on record is an unretried miss, and ep9 proves a second
approach converts; (2) re-plan at the close (`n_action_steps` 100 → ~25,
inference-time, testable on existing policies via `offline_eval.py` NSTEP);
(3) v2 data recipes — episode-constant small geometry, no occluders on the
nominal line, 67/33 stock-anchored mix, recovery demos in recollection. The
v2 protocol implementing all of this is `docs/RECIPE_V2.md`.

---

## 12. Artifact catalog (completed work)

### 12.0 Policy summary

All policies are ACT (~52M parameters, 206,699,768 bytes ≈ 200 MB), trained
with the identical recipe: 100,000 target steps, batch 8, seed 7, 4 dataloader
workers, prefetch 4, lr 1.0e-05, PyAV video backend, checkpoints every 20k,
no Hub push during training. Inputs: 6D state, front RGB 640×480, wrist RGB
1280×720 at 30 FPS; output: 6D actions.

| Policy | Steps | Final loss (l1/kld) | Grad norm | Steps/s | Train data | Model SHA-256 (first 12) |
|---|---|---|---|---|---|---|
| stock | 100k ✅ | 0.053 (0.052/0.000) | 4.167 | ~6.49 | 40 eps / 26,021 frames | `6718be1a97c1` |
| spatial | 100k ✅ | 0.061 (0.060/0.000) | 4.137 | ~6.42 | 360 eps / 234,189 frames | `5325ebc2af91` |
| occlusion | 100k ✅ | 0.064 (0.064/0.000) | 4.614 | ~6.20 | 360 eps / 234,189 frames | `f476c84bf04d` |
| full | 100k ✅ | 0.068 (0.068/0.000) | 5.097 | ~6.44 | 360 eps / 234,189 frames | `b961b703cb15` |
| lighting | 20k ⚠️ partial | ~0.16 at cancel | — | ~1.8–2.3 | 360 eps / 234,189 frames | `c2f1cb1711f9` |
| noise | cancelled ~15k ❌ | ~0.19 at cancel | — | ~1.1–1.7 | 360 eps / 234,189 frames | none — lost |

Loss interpretation: augmented buckets train on a deliberately harder, wider
distribution with the same 100k budget, so slightly higher final loss than
stock is expected and is not by itself a quality signal. The experiment's
metric is physical rollout success, not training loss.

Repositories (all private):

| Policy | Model repo | Dataset repo |
|---|---|---|
| stock | `Dillonjohnson/pick_place_v3_act_stock` | `Dillonjohnson/pick_place_v3_stock` |
| spatial | `Dillonjohnson/pick_place_v3_act_spatial` | `Dillonjohnson/pick_place_v3_spatial` |
| occlusion | `Dillonjohnson/pick_place_v3_act_occlusion` | `Dillonjohnson/pick_place_v3_occlusion` |
| full | `Dillonjohnson/pick_place_v3_act_full` | `Dillonjohnson/pick_place_v3_full` |
| lighting | `Dillonjohnson/pick_place_v3_act_lighting` (partial) | not uploaded |
| noise | `Dillonjohnson/pick_place_v3_act_noise` (empty) | not uploaded |

Collection:
`https://huggingface.co/collections/Dillonjohnson/so101-pick-place-v3-act-policies-6a74e56a093a15c8acc55786`

Using any completed policy (NUC or anywhere with HF access):

```bash
huggingface-cli login   # once, with a token that can read private repos
lerobot-rollout ... --policy.path=Dillonjohnson/pick_place_v3_act_stock
# or materialize locally:
huggingface-cli download Dillonjohnson/pick_place_v3_act_stock \
  --local-dir /home/multiply/LMFAO/policies/stock
```

Every completed policy repo root contains the full deployable set:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
training_summary.json
README.md
```

Each completed dataset repo has 1,085 data files (720 MP4 + 362 parquet +
meta), byte-verified against its training pod; stock's native-sharded dataset
has 9 data files. Dataset sizes on HF: stock ~0.62 GB, spatial ~8.5 GB,
occlusion ~7.1 GB, full ~15.0 GB.

### 12.1 Stock policy + dataset

```text
pod:          lmfao-act-stock / fnre6t63nr42wq  (EXITED after verify)
done marker:  /workspace/runs/eval_buckets/stock.done
checkpoints:  020000, 040000, 060000, 080000, 100000
last:         100000
```

Metrics (`training_summary.json`):

```text
final loss:         0.053
final l1 / kld:     0.052 / 0.000
final grad norm:    4.167
final lr:           1.0e-05
final mem_gb:       8.42
throughput:         ~6.49 steps/s
episodes / frames:  40 / 26,021
```

Policy repo `Dillonjohnson/pick_place_v3_act_stock`:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
training_summary.json
README.md
```

```text
model size:   206,699,768 bytes
model SHA-256:
6718be1a97c1b644884d7038ee0d3cc611ae445a69cedd815506dc1e2ed68e0b
```

Dataset repo `Dillonjohnson/pick_place_v3_stock` (9 data files + README /
`.gitattributes`; all sizes matched the pod exactly):

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.front/chunk-000/file-000.mp4
videos/observation.images.front/chunk-000/file-001.mp4
videos/observation.images.wrist/chunk-000/file-000.mp4
videos/observation.images.wrist/chunk-000/file-001.mp4
```

### 12.2 Spatial policy + dataset

```text
pod:          lmfao-act-spatial / q7k4rlc4d5cp0w  (EXITED after verify)
done marker:  /workspace/runs/eval_buckets/spatial.done
checkpoints:  020000, 040000, 060000, 080000, 100000
last:         100000
dataset path: /workspace/eval_buckets/spatial
```

Metrics:

```text
final loss:         0.061
final l1 / kld:     0.060 / 0.000
final grad norm:    4.137
final lr:           1.0e-05
final mem_gb:       8.43
throughput:         ~6.42 steps/s
episodes / frames:  360 / 234,189
```

Policy repo `Dillonjohnson/pick_place_v3_act_spatial` — same deployable file
set as stock.

```text
model size:   206,699,768 bytes
model SHA-256:
5325ebc2af91cf565cc50325d0036cb626b97f224e63f1bae426defab38f248c
```

Dataset repo `Dillonjohnson/pick_place_v3_spatial`:

```text
remote files:           1087 (1085 data + README + .gitattributes)
data files:             1085
mp4 / parquet:          720 / 362
bytes vs pod:           8,519,510,347 exact match
size mismatches:        0
upload note:            interrupted at 502 files, resumed to completion
```

### 12.3 Occlusion policy + dataset

```text
pod:          lmfao-act-occlusion / qmpthutzu6r2dl  (EXITED after verify)
done marker:  /workspace/runs/eval_buckets/occlusion.done
checkpoints:  020000, 040000, 060000, 080000, 100000
last:         100000
dataset path: /workspace/eval_buckets/occlusion
```

Metrics:

```text
final loss:         0.064
final l1 / kld:     0.064 / 0.000
final grad norm:    4.614
final lr:           1.0e-05
final mem_gb:       8.43
throughput:         ~6.2 steps/s
episodes / frames:  360 / 234,189
wall time:          ~4h 29m
```

Policy repo `Dillonjohnson/pick_place_v3_act_occlusion` — same deployable file
set as stock.

```text
model size:   206,699,768 bytes
model SHA-256:
f476c84bf04df3daaa346a975ebed68781a0d54056751b742d0b09086c38dbe8
```

Dataset repo `Dillonjohnson/pick_place_v3_occlusion`:

```text
remote files:           1087 (1085 data + README + .gitattributes)
data files:             1085
mp4 / parquet:          720 / 362
bytes vs pod:           7,131,537,676 exact match
size mismatches:        0
```

### 12.4 Full policy + dataset

```text
pod:          lmfao-act-full / q5vlgbs5tnxsyd  (EXITED after verify)
done marker:  /workspace/runs/eval_buckets/full.done
checkpoints:  020000, 040000, 060000, 080000, 100000
last:         100000
dataset path: /workspace/eval_buckets/full
```

Metrics:

```text
final loss:         0.068
final l1 / kld:     0.068 / 0.000
final grad norm:    5.097
final lr:           1.0e-05
final mem_gb:       8.43
throughput:         ~6.44 steps/s
episodes / frames:  360 / 234,189
wall time:          ~4h 19m
```

Policy repo `Dillonjohnson/pick_place_v3_act_full` — same deployable file
set as stock.

```text
model size:   206,699,768 bytes
model SHA-256:
b961b703cb1538ce563b0f79a1fdb495a0a32b9d5bafe2c4cd9c6e6e3950c552
```

Dataset repo `Dillonjohnson/pick_place_v3_full`:

```text
remote files:           1087 (1085 data + README + .gitattributes)
data files:             1085
mp4 / parquet:          720 / 362
bytes vs pod:           14,977,326,251 exact match
size mismatches:        0
```

Note: the full dataset is roughly 15 GB — about 1.75× the other augmented
buckets. The combined augmentation stack (noise + lighting + crop + occlusion)
makes the video less compressible.

### 12.5 Lighting partial checkpoint (20k/100k) and noise (lost)

Lighting was cancelled at ~21k steps. Its last durable checkpoint (20,000) was
saved before teardown:

```text
repo:         Dillonjohnson/pick_place_v3_act_lighting  (private)
contents:     full deployable-format root + training_summary.json
status:       PARTIAL — 20,000 / 100,000 steps
model size:   206,699,768 bytes
model SHA-256:
c2f1cb1711f9986ca4f2761709bf13627b55d64478660797ed567824ece9e63e
loss at save: ~0.16 (still descending; undertrained)
```

This checkpoint is deployable in format but is **not** comparable to the 100k
policies. Do not include it in the matched evaluation without retraining.

Noise was cancelled at ~15k steps, before its first 20k checkpoint was
written. Nothing recoverable exists; `Dillonjohnson/pick_place_v3_act_noise`
remains an empty reserved repo.

Neither lighting nor noise dataset was uploaded to HF. The original long-GOP
buckets remain on the NUC at `/home/multiply/LMFAO/eval_buckets_v3`; the
low-GOP re-encodes existed only on the deleted pods and would need to be
regenerated with `scripts/reencode_low_gop.py` for any retrain.

---

## 13. Lessons learned (do not pay these costs again)

### Diagnose random access before scaling trainers

Transfer pain was real, but long GOPs were the training killer. Benchmark one
dataset item and one training step before provisioning the fleet. After a short
warmup, enforce a throughput gate near the stock baseline.

### Encode for random access from the start

Next time, emit training videos with a short GOP (or re-encode before any ACT
job). Do not discover a 250-frame GOP after six pods are burning money.

### Never rename or mutate a live training pod

Rename restarted containers, broke ephemeral Python symlinks, forced job
resumes, and discarded uncheckpointed stock progress. Choose final names
before launch.

### Never kill active training without explicit approval

Slow ≠ stoppable. Lighting/noise stayed slow after low-GOP; observe them.
Stop only verified-idle pods after HF retention.

### Prepare one reusable CUDA environment

Per-pod git installs after disruption wasted hours. Clone a proven Python/CUDA
tree or bake a template image first.

### Stream augmentation; preserve both cameras; match LeRobot v3 exactly

OOM, silent front-only export, and schema mismatches all blocked the path
before cloud training. Smoke the full reader→augment→write→ACT-load loop on a
tiny set first.

### Keep monitoring scoped

Old transfer/stock monitors kept ticking after their jobs finished. Kill
obsolete loops; keep one clear handoff monitor.

### FFmpeg automation details that bit us

- NVENC advertised ≠ NVENC working; validate one encode first.
- Always `ffmpeg -nostdin` in SSH heredocs.
- Temp files must end in `.mp4` (`*.low-gop.tmp.mp4`) and be excluded from
  resume discovery.

---

## 14. Recommended sequence for the next experiment

1. Record and review source demos.
2. Run a tiny dual-camera augmentation + ACT smoke.
3. Validate schema and both cameras.
4. Encode (or re-encode) training videos with a random-access-friendly GOP
   **before** fleet training.
5. Benchmark 20–30 s of ACT on one augmented bucket; require near-stock
   throughput.
6. Build one validated Python/CUDA environment; snapshot/clone it.
7. Provision pods with final canonical names.
8. Transfer with resumable `rsync`; verify bytes and file counts.
9. Validate schema, cameras, CUDA, and one dataset sample on each pod.
10. Launch all policies with identical hyperparameters.
11. Checkpoint often enough to bound restart loss (20k worked here).
12. On `.done`: upload policy (+ dataset if retaining), verify SHA/bytes,
    then stop only that idle pod.
13. Roll out all policies on the same robot/camera setup and score physical
    success.
14. Terminate volumes only after durable retention is confirmed.

---

## 15. Security requirements

- Never put API keys, private keys, tokens, or `.env` files in this repository.
- Never paste private-key contents into commands, chat, or documentation.
- Never use `git add -A`, `git add .`, or `git add -u` in this public repo.
- Stage only explicit reviewed paths; run the secret scan before commit.
- Treat any leaked key as compromised and rotate it.
- Ephemeral pod-to-pod keys belong outside the repo and should be removed after
  transfer.
- HF tokens for uploads stay on the Mac and are passed to pods only in memory.

---

## 16. Relevant repository files

Core pipeline:

```text
scripts/record_demos.sh
scripts/augment_eval_buckets.sh
scripts/train_eval_buckets.sh
scripts/train_act.sh
scripts/reencode_low_gop.py
configs/training/buckets/
```

Augmentation implementation:

```text
src/lmfao/cli.py
src/lmfao/program.py
src/lmfao/replay.py
src/lmfao/datasets/lerobot.py
src/lmfao/datasets/episode.py
src/lmfao/features/spatial/random_crop.py
src/lmfao/features/spatial/utils.py
```

Regression tests:

```text
tests/test_program.py
tests/test_multicam.py
```

SO101 tooling:

```text
so101/README.md
so101/env.sh
so101/setup.sh
so101/rec
so101/guided_record.py
so101/teleop.sh
so101/eval/cam_check.py
so101/eval/live_view.py
so101/eval/read_pose.py
so101/eval/ring_spot.py      # v2 constants — stale for v3, superseded by preflight
```

Rollout / evaluation harness (added 2026-08-06 alongside the stock baseline):

```text
so101/series                 # N recorded trials, gated
so101/trial                  # one trial (delegates to series)
so101/eval_py                # runs an eval script under the venv + dialout
so101/eval/preflight.py      # weights, cameras, holders, pose, lighting, puck
so101/eval/goto_start.py     # drive to the demos' mean start pose, hold torque
so101/eval/review.py         # score a run: did the puck reach the box
```

Related docs:

```text
docs/STATUS.md
docs/TRAINING_RUN.md
docs/policytraining_v3.md
```

---

## 17. Definition of done

**Training phase final state (2026-08-06):**

- stock, spatial, occlusion, full reached 100,000 steps, each with a valid
  `checkpoints/last/pretrained_model`, uploaded to its private HF model repo
  with verified model size and LFS SHA-256, plus byte-verified datasets
- lighting was cancelled at ~21k; its 20k checkpoint is retained on HF as a
  partial artifact
- noise was cancelled at ~15k before its first checkpoint; nothing retained
- all RunPod pods deleted; no cloud billing remains

**Experiment complete when:**

- the four completed policies are rolled out on the same robot/camera setup
- nominal and held-out trials are recorded
- success rates and failure modes are compared
- RunPod resources are cleaned up after durable retention
