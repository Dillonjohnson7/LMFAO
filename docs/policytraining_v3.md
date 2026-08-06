# Policy Training v3: complete data-to-rollout runbook

Last updated: 2026-08-06

This is the detailed operational record for the SO101 `pick_place_v3`
experiment: collect fresh demonstrations, build five LMFAO augmentation
buckets, validate LeRobot v3 compatibility, train six matched ACT policies on
RunPod, and roll the policies out on the NUC-connected robot.

This document intentionally does not contain API tokens, private keys, or
private-key contents. RunPod addresses and forwarded SSH ports are ephemeral;
always confirm them in the RunPod console before using an example command.

## 1. Immediate next step: roll out the completed stock policy on the NUC

The stock policy finished all 100,000 ACT training steps. Its deployable
artifact is:

```text
/workspace/runs/eval_buckets/stock/checkpoints/last/pretrained_model
```

On the stock RunPod, `checkpoints/last` currently points to `100000`. The
deployable directory is about 200 MB and contains:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

Do not copy only `model.safetensors`. LeRobot needs the complete
`pretrained_model` directory, including the normalization processors and
configuration.

### 1.1 Pull this repository update onto the NUC

This file must first be committed and pushed from the development machine.
After that:

```bash
ssh multiply@100.103.79.98
cd /home/multiply/LMFAO
git pull
less docs/policytraining_v3.md
```

Writing this file does not itself publish it. A separate explicit commit and
push are required before `git pull` can retrieve it on the NUC.

### 1.2 Copy the stock policy from RunPod or Hugging Face to the NUC

Preferred durable source after extraction:

```text
Policy:  Dillonjohnson/pick_place_v3_act_stock
Dataset: Dillonjohnson/pick_place_v3_stock
```

On the NUC, either download from Hugging Face after `huggingface-cli login`, or
copy directly from the still-running stock pod:

```bash
cd /home/multiply/LMFAO
mkdir -p policies

# Confirm the current public IP and forwarded SSH port in RunPod first.
# Snapshot on 2026-08-06: stock was 209.170.80.132, forwarded port 14132.
STOCK_HOST=209.170.80.132
STOCK_PORT=14132

scp -r \
  -i "$HOME/.ssh/lmfao_runpod_transfer" \
  -P "$STOCK_PORT" \
  "root@${STOCK_HOST}:/workspace/runs/eval_buckets/stock/checkpoints/last/pretrained_model" \
  /home/multiply/LMFAO/policies/stock
```

If that key is rejected, its public half must be added to the stock pod's
`/root/.ssh/authorized_keys`. Never copy the private key into this repository
or paste it into documentation.

On 2026-08-06, the NUC transfer public key was added to the stock pod and a
NUC-to-RunPod check returned `NUC_ACCESS_OK`. The private key remained on the
NUC. This authorization is container-local and may need to be repeated after a
future stock-pod restart.

Validate the download:

```bash
cd /home/multiply/LMFAO
test -f policies/stock/model.safetensors
test -f policies/stock/config.json
test -f policies/stock/policy_preprocessor.json
du -sh policies/stock
```

Expected size is approximately 200 MB.

### 1.3 Verify the robot and cameras before policy control

The trained stock policy expects exactly:

- `observation.state`: 6 values
- `observation.images.front`: RGB, 640×480
- `observation.images.wrist`: RGB, 1280×720
- `action`: 6 values
- 30 FPS

The current NUC camera pins were:

```text
front: /dev/video6
wrist: /dev/video0
```

Pins can change after USB replugging. Verify rather than trusting the snapshot:

```bash
cd /home/multiply/LMFAO
source so101/env.sh

python so101/eval/cam_check.py
python so101/eval/live_view.py
python so101/eval/read_pose.py 3
```

If needed, re-pin:

```bash
python so101/eval/cam_check.py --find-front
python so101/eval/cam_check.py --find
```

Also verify:

```bash
readlink -f /dev/so101_follower
cat so101/front_cam.path
cat so101/wrist_cam.path
```

The policy was trained with both cameras. Do not use `WRIST=0`, swap camera
names, or feed both names from the same UVC device.

### 1.4 First rollout: five-second safety smoke

Safety:

- Clear people and objects from the arm's swept volume.
- Keep an immediate power-disconnect or emergency-stop action available.
- Start with the puck and box in the same nominal training layout.
- Do not connect the leader arm for a base policy rollout.
- Start at five seconds, then increase only after motion looks sane.
- The follower energizes on connection. Expect torque to engage.

Run from the NUC:

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

`lerobot-record` is not the policy deployment command in the installed LeRobot
version. Its own validation says to use `lerobot-rollout` for policy-based
deployment.

The NUC has no CUDA GPU, so both `--policy.device=cpu` and `--device=cpu` are
explicit. If CPU inference cannot sustain the control rate, lower the rollout
FPS only as a deliberate diagnostic; do not silently change the comparative
evaluation protocol.

If the five-second smoke is safe, repeat at 20 seconds:

```bash
# Use the same command, changing only:
--duration=20
```

The base strategy does not record an evaluation dataset. For the first physical
test, manually record:

- policy name and checkpoint
- date/time
- scene condition
- success/failure
- failure mode
- whether a human stopped the rollout

## 2. Experiment goal

The experiment asks whether LMFAO augmentation improves physical SO101
pick-and-place robustness. It is not a loss-only benchmark and not a comparison
against the old `pick_place_v2` smoke dataset.

The controlled design is:

- Recollect one fresh 40-episode stock dataset.
- Keep ACT hyperparameters and seed identical.
- Change only the training dataset.
- Train six policies: stock, lighting, noise, occlusion, spatial, and full.
- Compare physical rollout success in both nominal and held-out conditions.

The six datasets are:

- `stock`: 40 untouched demonstrations.
- `lighting`: originals plus brightness, contrast, and color-temperature
  variants.
- `noise`: originals plus Gaussian and uniform sensor-noise variants.
- `occlusion`: originals plus border, sequence-box, and moving-box variants.
- `spatial`: originals plus random-crop variants.
- `full`: originals plus a combined lighting, Gaussian-noise, border-occlusion,
  and random-crop pipeline.

Each augmented bucket contains:

- 40 source episodes
- 8 augmented variants per source
- the original source retained with `--include-original`
- 360 total episodes
- 234,189 total frames
- two camera streams
- 720 MP4 files after LMFAO export

The stock bucket contains 40 episodes and 26,021 frames.

Within each augmented bucket, episodes are grouped by source: the exact
original comes first, followed by variants 0–7. Originals are therefore about
11% of the bucket and augmented examples about 89%.

The stock recorder uses native LeRobot video sharding, so its 40 episodes are
packed into two shared MP4 shards per camera. An early check incorrectly
expected one file per episode; the smaller stock file count was normal and not
data loss.

## 3. Machine boundaries

The workflow deliberately spans three machine classes.

### Development Mac

Responsibilities:

- source-code changes
- tests
- RunPod provisioning and status checks
- orchestration

The Mac must not be used as the persistent staging point for multi-gigabyte
datasets or training environments. It must never store RunPod private keys in
the repository.

### SO101 NUC / robot workstation

Identity used during this run:

```text
multiply@100.103.79.98
```

Responsibilities:

- physical teleoperation
- fresh data collection
- CPU augmentation
- episode review
- physical policy rollout

Important paths:

```text
/home/multiply/LMFAO
/home/multiply/envs/lerobot06
/home/multiply/LMFAO/eval_buckets_v3
/home/multiply/LMFAO/so101/datasets/pick_place_v3
```

### RunPod CUDA machines

Responsibilities:

- GPU ACT training
- temporary bucket storage
- training checkpoints

Canonical pod names after cleanup:

```text
lmfao-act-stock
lmfao-act-lighting
lmfao-act-noise
lmfao-act-occlusion
lmfao-act-spatial
lmfao-act-full
```

Public IPs and forwarded ports are not stable identifiers. Pod names and IDs
are more durable, but even pods should be treated as temporary. Copy completed
policy artifacts before deleting a pod.

## 4. Fresh data collection on the NUC

The production evaluation uses `pick_place_v3`, not the old
`Dillonjohnson/pick_place_v2` smoke dataset.

Record:

```bash
cd /home/multiply/LMFAO
./scripts/record_demos.sh 40
# Equivalent:
./so101/rec 40
```

Default task:

```text
pick the blue puck and place it in the brown box
```

Collection protocol:

1. Keep lighting, camera mounts, and table scene fixed.
2. Put the puck in the nominal start area.
3. Match leader and follower poses before connecting.
4. Teleoperate slowly and consistently.
5. Press ENTER to stop each recording.
6. Keep good episodes, redo faults, and do not save correction episodes.
7. Keep both front and wrist cameras enabled.

The guided recorder rejects empty/faulted episodes and can resume an existing
dataset only when its observation camera set matches.

## 5. Augmentation design and exact configs

The production command:

```bash
cd /home/multiply/LMFAO
export STOCK=/home/multiply/LMFAO/so101/datasets/pick_place_v3
export OUT_ROOT=/home/multiply/LMFAO/eval_buckets_v3
export VARIANTS=8
export SEED=7
./scripts/augment_eval_buckets.sh
```

The script creates `stock` as a symlink to the fresh source and runs
`lmfao augment` with:

```text
--variants 8
--include-original
--seed 7
--resume
```

Exact bucket configurations live under `configs/training/buckets/`.

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

Full:

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

The `full` config is a sequential combined pipeline, not a set of disjoint
families. A full-bucket variant receives brightness, contrast, color
temperature, Gaussian noise, and random crop in the same generated video. The
border occlusion is additionally applied with probability 0.5.

## 6. Production fixes made during this run

### 6.0 Smoke validation before production

Before collecting the 40 production demos, three NUC smoke demonstrations were
used to test the end-to-end reader, augmenter, LeRobot writer, and ACT loader.

That smoke exposed two critical facts:

- the original augmentation path could silently emit front-only output from a
  dual-camera source
- the exported LeRobot schema had to match v3 exactly before ACT would load it

The CLI was first hardened to reject ambiguous multi-camera input unless the
video key was explicit, then extended to preserve and replay augmentation onto
both cameras. A GPU ACT smoke subsequently loaded the augmented data and
reduced loss from roughly 74 to 22 over ten steps, proving trainability before
the production run.

### 6.1 Bounded-memory augmentation

The initial implementation could materialize every variant of a high-resolution
dual-camera episode at once. One uncompressed episode can occupy several GiB,
so eight variants could exhaust NUC RAM.

This was observed in production: Linux killed the first lighting augmentation
process under memory pressure after it had produced most of that bucket. The
source data remained safe, but the run had to be resumed after the generator
refactor.

The production path was changed to generators:

- `iter_augment_episode`
- `iter_sweep_episode`

The CLI now yields and writes one produced episode at a time. Batch APIs still
exist, and tests verify that streamed output preserves batch order, metadata,
frames, and deterministic seed behavior.

Relevant files:

```text
src/lmfao/program.py
src/lmfao/cli.py
tests/test_program.py
```

### 6.2 Dual-camera augmentation replay

LMFAO applies the selected transform history to the secondary camera so front
and wrist retain the same geometric/photometric augmentation intent.

A bug treated Gaussian and uniform noise as if both stored `sigma`. In reality:

- Gaussian noise records `sigma`.
- Uniform noise records `amplitude`.

`src/lmfao/replay.py` now handles those parameter names separately. A
multi-camera regression test covers both noise types.

Relevant files:

```text
src/lmfao/replay.py
tests/test_multicam.py
```

### 6.3 LeRobot v3 schema compatibility

ACT smoke training exposed strict LeRobot v3 requirements. The writers now
emit:

1. `meta/stats.json` and per-episode feature statistics.
2. `tasks.parquet` with the expected pandas index metadata.
3. `info.json` declarations for every data column, including state, action,
   videos, timestamps, frame/episode/index fields, task index, splits, and size
   fields.
4. State and action as `fixed_size_list<float32>[dim]`, not variable lists.

Both camera streams were validated after production augmentation.

### 6.4 Training output directory safety

LeRobot requires a new `output_dir` when `resume=false`. Pre-creating the final
run directory caused `FileExistsError`.

The scripts now:

- fail if the final output already exists
- create only its parent directory
- let `lerobot-train` create the final run directory

Relevant files:

```text
scripts/train_act.sh
scripts/train_eval_buckets.sh
```

## 7. Validation performed before cloud training

Each production bucket was checked for:

- complete LeRobot metadata
- expected 360 episodes
- expected 234,189 frames
- both `observation.images.front` and `observation.images.wrist`
- state/action dimensions
- video existence and decode
- no macOS AppleDouble `._*` files
- CUDA availability on the target training pod

The old `pick_place_v2` dataset was used only for smoke validation. It was not
used as the comparative production dataset.

At the user's request, two original stock video shards were also copied for
manual viewing during augmentation:

```text
~/Downloads/lmfao_stock_samples/stock_front.mp4
~/Downloads/lmfao_stock_samples/stock_wrist_top.mp4
```

They were approximately 100 MB and 98 MB and contain multiple stock episodes.
They are historical review artifacts, not training inputs and not required for
rollout.

## 8. Dataset transfer history and lessons

The transfer process went through several iterations:

1. Direct tar streams were attempted.
2. File-by-file resumable Python transfer was attempted.
3. `rsync` with `--partial --append-verify` became the robust NUC-to-pod method.
4. `full` and `occlusion` staging used direct pod-to-pod transfer to avoid
   retransmitting through the Mac.

The robust pattern was:

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

Important transfer lessons:

- Use resumable transfer for multi-gigabyte buckets.
- Validate file count and byte count after transfer.
- Avoid routing large data through the Mac.
- macOS tar can create `._*` AppleDouble files; use `COPYFILE_DISABLE=1` if a
  Mac-originated tar is unavoidable.
- Ephemeral RunPod SSH endpoints may change after a restart.
- A pod name update can restart the container and change its forwarded port.

## 9. The training-speed failure and real root cause

Initial augmented training was dramatically slower than stock. Increasing data
loader workers was considered, but transfer speed and worker count were not the
root cause.

A more balanced loader configuration temporarily improved spatial training to
roughly 2–3 steps/second, but this remained well below the eventual 6.4–6.6
steps/second range. TorchCodec was also evaluated as an alternate video
backend. The durable fix remained correcting the source videos' random-access
structure and using the validated PyAV path consistently.

The augmented MP4 files had long H.264 GOPs:

```text
keyframe gaps: approximately 250 frames
```

ACT performs random frame access. PyAV must seek to the previous keyframe and
decode forward. With a 250-frame GOP, each random sample could decode seconds of
unneeded video, leaving the GPU starved.

A direct random-access benchmark on one production clip measured:

```text
original long-GOP video: 0.33 independent frame requests/second
low-GOP video:           2.76 independent frame requests/second
improvement:             approximately 8.4×
```

The correction was to re-encode augmented videos with:

```text
codec: libx264
preset: veryfast
CRF: 18
GOP: 2
minimum keyframe interval: 2
scene-cut keyframes: disabled
pixel format: yuv420p
```

The resumable utility is:

```text
scripts/reencode_low_gop.py
```

It:

- scans all MP4s
- skips files already using a two-frame GOP
- writes a temporary MP4 beside the source
- validates frame/packet count and keyframe spacing
- atomically replaces the source only after validation
- removes failed temporary output
- supports multiple worker processes
- excludes its own `*.low-gop.tmp.mp4` files from discovery

The original NUC datasets were not modified. Re-encoding occurred on each
RunPod copy.

After the fix, measured ACT throughput was:

```text
stock resume:     approximately 6.6 steps/second
spatial:          approximately 6.43 steps/second
occlusion:        approximately 6.33 steps/second
full:             approximately 6.45 steps/second
```

This confirmed that long-GOP video seeking—not the ACT model or GPU—was the
primary augmented-training bottleneck.

## 10. ACT training configuration

All policies use the same comparison settings:

```text
policy: ACT
steps: 100000
batch size: 8
data loader workers: 4
prefetch factor: 4
seed: 7
device: CUDA
push to Hub: false
checkpoint saving: true
checkpoint frequency: 20000
log frequency: 200
Weights & Biases: disabled
video backend: PyAV
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

The ACT model has approximately 52 million parameters. The stock deployable
checkpoint is approximately 200 MB.

## 11. RunPod environment

The known-working environment used:

```text
Python:      3.12.13
LeRobot:     0.6.2 from pinned git revision
revision:    ef88d4e52b9f3a16638e4b73202d619b7606fd41
Torch:       2.11.0+cu128
Torchvision: 0.26.0+cu128
PyAV:        15.1.0
TorchCodec:  0.15.0
Accelerate:  1.14.0
Datasets:    5.0.1
```

LeRobot at this revision requires Python 3.12 or newer and Torch 2.7 or newer.
The full pod's base image had Python 3.11 and Torch 2.4, so it was not a valid
substitute.

PyAV must remain on the 15.x line for this LeRobot setup. Newer incompatible
versions can break APIs used by LeRobot.

## 12. Current run status snapshot

Snapshot taken around 2026-08-06 18:24 America/New_York:

- `stock`: complete at 100,000/100,000. Final checkpoint, training summary, and
  stock LeRobot dataset extracted to private Hugging Face repositories. Pod
  `lmfao-act-stock` remains RUNNING with SSH access.
- `spatial`: complete at 100,000/100,000. Final checkpoint present at
  `checkpoints/last -> 100000` with full `pretrained_model` contents. Measured
  finish throughput ~6.42 steps/second. Not yet uploaded to Hugging Face in
  this snapshot.
- `occlusion`: approximately 92,920/100,000 at ~4–6 steps/second.
- `full`: approximately 73,130/100,000 at ~6.5 steps/second.
- `lighting`: approximately 11,963/100,000 at ~2.2–2.4 steps/second.
- `noise`: approximately 8,655/100,000 at ~1.1–1.3 steps/second.

NUC physical rollouts are being handled separately by the operator and are not
tracked as live status in this snapshot.

These counters are a time-stamped snapshot, not a live status API. Read the
current pod logs before making billing or shutdown decisions.

Run paths:

```text
/workspace/runs/eval_buckets/stock
/workspace/runs/eval_buckets/spatial
/workspace/runs/eval_buckets/occlusion
/workspace/runs/eval_buckets/full
/workspace/runs/eval_buckets/lighting
/workspace/runs/eval_buckets/noise
```

Logs and markers:

```text
/workspace/runs/eval_buckets/BUCKET.log
/workspace/runs/eval_buckets/BUCKET.launch.log
/workspace/runs/eval_buckets/BUCKET.pid
/workspace/runs/eval_buckets/BUCKET.done
/workspace/runs/eval_buckets/BUCKET.failed
```

### 12.1 Hugging Face retention layout

All six policies have separate private Hugging Face model repositories grouped
in one private v3 collection. Hugging Face does not support nested paths such
as `Dillonjohnson/v3/stock`, so the collection is the folder-like grouping while
each model remains directly usable as `--policy.path`.

```text
Collection:
https://huggingface.co/collections/Dillonjohnson/so101-pick-place-v3-act-policies-6a74e56a093a15c8acc55786

Policy repositories:
Dillonjohnson/pick_place_v3_act_stock
Dillonjohnson/pick_place_v3_act_lighting
Dillonjohnson/pick_place_v3_act_noise
Dillonjohnson/pick_place_v3_act_occlusion
Dillonjohnson/pick_place_v3_act_spatial
Dillonjohnson/pick_place_v3_act_full

Stock dataset repository:
Dillonjohnson/pick_place_v3_stock
```

Deployable policy files live at each model repository root. Credentials,
private keys, and intermediate incomplete checkpoints are not uploaded.

### 12.2 Stock extraction completed from `lmfao-act-stock`

The completed stock pod was inventoried and durable HF retention was verified.

Pod facts:

```text
pod name:     lmfao-act-stock
pod id:       fnre6t63nr42wq
status:       RUNNING
SSH:          209.170.80.132:14132
done marker:  /workspace/runs/eval_buckets/stock.done
last ckpt:    checkpoints/last -> 100000
checkpoints:  020000, 040000, 060000, 080000, 100000
```

Final training metrics from the stock log:

```text
steps:                 100000
batch size:            8
seed:                  7
num_workers:           4
prefetch_factor:       4
final loss:            0.053
final l1_loss:         0.052
final kld_loss:        0.000
final grad norm:       4.167
final lr:              1.0e-05
final mem_gb:          8.42
measured steps/sec:    ~6.49
```

Private policy repository contents:

```text
https://huggingface.co/Dillonjohnson/pick_place_v3_act_stock

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

Verified model artifact:

```text
model size:   206,699,768 bytes
model SHA-256:
6718be1a97c1b644884d7038ee0d3cc611ae445a69cedd815506dc1e2ed68e0b
```

Private stock dataset repository contents:

```text
https://huggingface.co/datasets/Dillonjohnson/pick_place_v3_stock

meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.front/chunk-000/file-000.mp4
videos/observation.images.front/chunk-000/file-001.mp4
videos/observation.images.wrist/chunk-000/file-000.mp4
videos/observation.images.wrist/chunk-000/file-001.mp4
README.md
```

Dataset facts extracted from the pod copy:

```text
episodes: 40
frames:   26,021
fps:      30
bytes:    ~619,687,715 on pod before upload
cameras:  front + wrist
```

NUC rollout can use either:

```bash
# local copy from the still-running stock pod
scp -r -i "$HOME/.ssh/lmfao_runpod_transfer" -P 14132 \
  root@209.170.80.132:/workspace/runs/eval_buckets/stock/checkpoints/last/pretrained_model \
  /home/multiply/LMFAO/policies/stock

# or Hugging Face after login on the NUC
# --policy.path=Dillonjohnson/pick_place_v3_act_stock
```

Remaining policies should follow the same extraction pattern after they reach
100,000 steps: validate `checkpoints/last/pretrained_model`, upload the complete
deployable folder, write `training_summary.json`, and keep intermediate
checkpoints off Hugging Face unless explicitly requested.

## 13. Significant failures, mistakes, and recovery

This section is intentionally explicit so the same time and cost are not spent
again.

### Transfer was blamed too early

Slow transfer was initially treated as the main schedule risk. Transfer did
need hardening, but it was not the reason augmented ACT steps were extremely
slow. The video keyframe structure should have been measured earlier.

Lesson: benchmark one dataset item and one training step before scaling out.

### Training ran too long before isolating random access

The first augmented jobs were allowed to run at poor throughput before a
single-frame random-seek benchmark isolated the problem.

Lesson: after a short warmup, enforce a throughput gate. If an RTX-class pod is
far below the stock baseline, stop and profile immediately.

### GPU FFmpeg encoding assumption was wrong

The packaged FFmpeg advertised `h264_nvenc`, but opening an NVENC session on the
4090 failed with `unsupported device`. CPU `libx264` was then benchmarked and
used.

Lesson: test one complete output and validate keyframes before launching a
bucket-wide re-encode.

### FFmpeg consumed heredoc input

One test omitted `-nostdin`, so FFmpeg consumed the remainder of the remote
shell heredoc as interactive commands.

Lesson: always use `ffmpeg -nostdin` inside SSH heredoc automation.

### Temporary output extension was initially invalid

The first re-encoder temporary filename ended in `.tmp`, so PyAV could not
infer an output container.

Fix: use `*.low-gop.tmp.mp4` and explicitly pass MP4 format.

### Resume discovery initially included temporary MP4s

Because temporary files ended in `.mp4`, a resumed recursive scan could treat
them as source files.

Fix: explicitly exclude names ending in `.low-gop.tmp.mp4`.

### Renaming live RunPods restarted containers

Aligning pod names triggered container restarts and changed forwarded SSH ports.
All in-container processes stopped, although `/workspace` data survived.

Consequences:

- re-encoding jobs had to be resumed
- Python interpreter symlinks pointing into ephemeral `/root` broke
- stock had reached roughly 88,703 steps but resumed from its last durable
  80,000 checkpoint, losing roughly 8,703 uncheckpointed steps

Lesson: never rename, resize, or otherwise mutate a live training pod. Choose
the final name before starting work.

### Full-pod environment rebuild took too long

The full pod's environment installation was interrupted by the rename.
Git-based installation stalled while fetching LeRobot. A later resolver started
pulling a complete CUDA stack. That was stopped.

Recovery:

- stage the pinned LeRobot source without storing it on the Mac
- clone the already-proven Python/CUDA environment directly from another
  RunPod
- validate imports, CUDA, LeRobot dataset loading, both cameras, and one dataset
  item
- launch full training

Lesson: prepare and validate a reusable RunPod image or persistent environment
before provisioning all training pods.

### Lighting and noise runtimes also broke after rename

The lighting and noise `/workspace/v312` virtual environments survived, but
their Python executable symlinks still targeted an ephemeral runtime under
`/root/.local/share/uv/python`. Re-encoding could run under system Python, so
this breakage was not visible until training validation.

Recovery copied only the known-working 110 MB Python 3.12.13 runtime from the
full pod to each affected `/workspace` volume and recreated the expected
symlink. No runtime or dataset was stored on the Mac.

### More workers did not fix lighting throughput

Lighting initially measured about 1.76 steps/second with four saturated loader
workers and low GPU utilization. It was restarted early with 16 workers and
prefetch factor 1, avoiding the previous high-prefetch OOM configuration, but
throughput remained about 1.9–2.3 steps/second.

Lighting and spatial have the same approximately 17.85-core CPU quota.
Therefore the remaining lighting slowdown is not a missing-worker or pod CPU
allocation problem. It is specific to lighting video decoding/storage; blindly
increasing workers again is not an appropriate fix.

### Monitoring loops became noisy

Old transfer and stock monitors continued emitting ticks after their work was
complete.

Fix: explicitly stop obsolete loops. Keep only the handoff monitor until
lighting and noise have launched.

## 14. Recommended repeatable production sequence

For the next experiment:

1. Record and review the source demos.
2. Run a small augmentation smoke.
3. Validate schema and both cameras.
4. Encode training videos with a random-access-friendly GOP from the start.
5. Benchmark 20–30 seconds of ACT throughput on one augmented bucket.
6. Require throughput near the stock baseline before provisioning all pods.
7. Build one validated Python/CUDA environment.
8. Snapshot it as a RunPod template or clone its persistent environment.
9. Provision pods with their final canonical names.
10. Transfer with resumable `rsync`.
11. Validate bytes, file count, schema, cameras, CUDA, and a dataset sample.
12. Start all policies with identical hyperparameters.
13. Save checkpoints frequently enough to limit restart loss.
14. Copy completed policy directories off temporary pods.
15. Stop/delete unused pods only after artifact verification.

## 15. Physical comparison plan

Training loss is not the experiment result. Use physical success rate.

Recommended conditions:

- nominal training layout
- dimmer lighting
- warmer lighting
- slight front-camera framing shift
- slight wrist-camera framing shift
- small edge occluder
- exposure/gain disturbance

Run the same number of trials per policy and condition. Twenty trials per
policy/condition is a reasonable first target.

For every trial record:

- policy bucket
- checkpoint step
- condition
- success/failure
- grasp success
- transport success
- placement success
- collision or unsafe motion
- human intervention
- short failure note

The augmentation family is useful when it beats stock in its matching held-out
condition without collapsing nominal success.

## 16. Artifact retention and pod cleanup

Before stopping a completed pod:

```bash
test -L /workspace/runs/eval_buckets/BUCKET/checkpoints/last
test -f /workspace/runs/eval_buckets/BUCKET/checkpoints/last/pretrained_model/model.safetensors
du -sh /workspace/runs/eval_buckets/BUCKET/checkpoints/last/pretrained_model
```

Copy the whole `pretrained_model` directory to durable storage. Then verify it
can be loaded on the destination before deleting the pod.

For this run, durable storage means the matching private Hugging Face model
repository. Verify the required files and LFS SHA-256 remotely before stopping
the pod.

Keep names aligned:

```text
lmfao-act-stock
lmfao-act-lighting
lmfao-act-noise
lmfao-act-occlusion
lmfao-act-spatial
lmfao-act-full
```

Do not retain obsolete `-retry`, `-ready`, or numbered pods after their
replacement is confirmed.

## 17. Security requirements

- Never put API keys, private keys, tokens, or `.env` files in this repository.
- Never paste private-key contents into commands, chat, or documentation.
- Never use `git add -A`, `git add .`, or `git add -u` in this public repo.
- Stage only explicit reviewed paths.
- Run the configured secret scan before commit.
- Treat any leaked key as compromised and rotate it.
- Ephemeral pod-to-pod keys belong outside the repository and should be removed
  after transfer.

## 18. Relevant repository files

Core pipeline:

```text
scripts/record_demos.sh
scripts/augment_eval_buckets.sh
scripts/train_eval_buckets.sh
scripts/train_act.sh
scripts/reencode_low_gop.py
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
so101/eval/ring_spot.py
```

Related documentation:

```text
docs/STATUS.md
docs/TRAINING_RUN.md
docs/policytraining_v3.md
```

## 19. Definition of done

The training phase is complete when:

- all six policies reach 100,000 steps
- each has a valid `checkpoints/last/pretrained_model`
- each artifact is uploaded to its matching private Hugging Face repository
- each repository contains the complete deployable root file structure
- model size and LFS SHA-256 are verified
- each artifact loads successfully

The experiment is complete when:

- all six policies are rolled out on the same robot/camera setup
- nominal and held-out trials are recorded
- success rates and failure modes are compared
- RunPod resources are cleaned up after artifact retention

