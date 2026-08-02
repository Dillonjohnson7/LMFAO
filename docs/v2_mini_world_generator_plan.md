# v2: Gaussian Splat Mini-World Generator (revised after literature review)

Status: **design doc / roadmap, not implemented**. Stretch goal referenced in the
project's v2 note ("MCTS → train"). Revised 2026-08-02 after checking the plan's
assumptions against published work; the original draft's claims that didn't hold up
are corrected inline rather than silently dropped (see §3 and §6).

**Naming note:** this is deliberately called a *mini-world generator*, not a "world
model." A world model in the RL/planning sense (Dreamer, MuZero) learns a predictive
dynamics function: given a state and action, it predicts what happens next. This
system predicts nothing: it deterministically **re-renders an already-recorded
trajectory** from new viewpoints/conditions using a fitted 3D reconstruction. It's a
scene generator/renderer, not a model of dynamics.

## 1. Why this isn't an `Augmenter`

Every feature module so far (`docs/adding_features.md`) implements
`Augmenter.apply(video, metadata, rng) -> (video, metadata)`: a stateless,
per-episode, per-frame transform. The mini-world generator doesn't fit that contract:
it needs many frames/episodes plus a one-time scene scan to fit a reconstruction,
fitting is itself an optimization process, and the strategy-selection loop (§6) sits
a level above data augmentation entirely.

Proposed shape: a new subsystem, `lmfao.miniworld`, that runs as an **optional,
switchable episode source in parallel with** the real-data path, not a stage the
real data passes through. A single toggle (`miniworld.enabled`) turns the branch on
or off; when off, the existing pipeline is byte-for-byte unaffected. When on, it
consumes raw episodes and **manufactures new synthetic episodes** in the same
LeRobot format (video + per-frame joint state); those merge with the real episodes
upstream of the pipeline, and `lighting.*` / `spatial.*` then season them like any
real episode:

```
raw episodes ----------------------------------------------+
                                                           +--> AugmentationPipeline -> training set
raw episodes --[switch]--> [miniworld: novel-view/scene- --+
   (copy)      on/off       edit synthesis] -> synthetic
                            episodes
```

It slots in *before* the existing pipeline, never *inside* it: the pipeline sees
one merged stream of episodes and neither knows nor cares which are synthetic
(provenance lives in metadata, §8d).

## 2. Validation from the literature: the core idea works, and better than we guessed

**RoboSplat** (OpenRobotLab, RSS 2025, arXiv:2504.13175) is almost exactly the
generation half of this plan, published and open-sourced (Apache-2.0): reconstruct
the scene as 3D Gaussians, segment robot/object Gaussians, then generate novel
demonstrations by editing the splat (object-pose transforms, appearance/object
swaps, lighting edits, novel camera views) from **a single expert demo**. Their
headline result: policies trained on hundreds of real demos + 2D augmentation
averaged **57.2%** success; RoboSplat's one-shot generated demos reached **87.8%**
across six generalization types. **RoboGSim** (arXiv:2411.11839) and **SplatSim**
corroborate the approach (RoboGSim handles the arm via Modified DH kinematics,
i.e. the same "render the arm from joint angles, don't reconstruct it" split this
plan proposed).

Two practical consequences:

- The expected payoff is not speculative: 2D-only augmentation (our current
  `lighting.*`/`spatial.*` modules) has a measured ceiling that scene-space
  generation demonstrably breaks through.
- **We should build on RoboSplat's released code**, not assemble
  gsplat/nerfstudio/COLMAP from scratch as the original draft proposed. That
  collapses most of Phase 1's estimated effort.

## 3. Corrections to the original draft

### 3a. Camera poses come from the robot, not from SfM

The original draft listed "COLMAP SfM on wrist-cam footage, unverified convergence"
as stage 1 and its top risk. Wrong tool: the wrist camera is **rigidly mounted to a
kinematic chain whose joint angles are recorded every frame**. A one-time hand-eye
calibration (standard; even Gaussian-splat-based variants exist, e.g. SurgCalib)
plus forward kinematics yields the wrist-cam pose for *every frame of every
episode* with no SfM at all. FK-derived poses can seed a pose-refinement step
during splat fitting if calibration error shows up as blur. COLMAP survives only as
an optional cross-check, not a dependency. The SO101's kinematic model is available
off the shelf (`TheRobotStudio/SO-ARM100` ships `so101_new_calib.urdf`;
`lerobot-kinematics` provides FK/IK for exactly this arm).

### 3b. Reconstruction source: existing demo footage first, scan pass only for future sessions

RoboSplat's pipeline takes "a single expert demonstration **and multi-view
images**"; they capture a deliberate multi-view scan. An earlier revision of this
plan copied that requirement. On reflection it should be inverted for our case:

- **Existing demo footage is likely sufficient.** The wrist cam sweeps a wide arc
  every episode (approach → descend → carry → bin), and `pick_place_v2` has 45
  episodes ≈ 31k frames with puck positions varied over ~18×18 cm, which gives
  thousands of distinct FK-posed viewpoints. Splat quality matters most *near the
  viewpoints we re-render from*, and our novel views are deliberately small offsets
  from the real cameras, so task-directed coverage covers exactly the region we
  need. Countermeasures for its weaknesses: filter to low-joint-velocity frames
  (kills motion blur; thousands of frames remain), mask the arm via FK-projected
  silhouette, and use only pre-grasp frames around the puck's region.
- **For already-recorded datasets it's the only option.** The July-21 scene no
  longer exists as filmed; a scan sweep done today would reconstruct a subtly
  different scene. Any augmentation of `pick_place_v2` must reconstruct from its
  own footage.
- **For future recording sessions**, append a 30-second scripted slow sweep to the
  recording protocol as cheap insurance (better background coverage, no motion
  blur), but treat it as protocol hygiene, not a pipeline dependency.

One-time hand-eye calibration is still required (§3a) and is also recoverable from
existing footage: run COLMAP on a few hundred wrist frames, align the resulting
trajectory to the FK gripper trajectory, and solve the standard AX=XB problem for
the fixed gripper→camera transform. COLMAP's role is this one-time calibration
bootstrap, not per-frame tracking.

### 3c. The puck is not static; it gets grasped and moved

The original draft put the puck in the "static background + rigid objects" splat.
Wrong for every frame after grasp. Correct handling (and what RoboSplat's
object-Gaussian segmentation supports): segment the puck's Gaussians; before grasp
it sits at its per-episode pose (blob-detect + triangulate, or annotate); after
grasp its pose is **rigidly attached to the gripper frame**, meaning gripper FK
composed with a fixed grasp transform. No tracking model needed during carry; the
phase boundary (grasp/release) is readable from the gripper-position channel
(README: carry ≈ gripper 26–36, missed ≈ <20).

### 3d. Beacons

Unchanged from the original draft, minus the puck correction above: gripper beacon
= pure FK from `observation.state`; puck beacon = pre-grasp pose estimate + grasp
attachment; both project into any rendered novel view for free.

## 4. Revised pipeline stages

| # | Stage | Tooling | Risk |
|---|---|---|---|
| 1 | One-time hand-eye calibration (COLMAP-on-footage + AX=XB vs. FK trajectory) | COLMAP + FK alignment; OpenCV target-based as alternative for future rigs | Low-medium; calibration accuracy directly bounds render sharpness. |
| 2 | Static-scene splat fit from velocity-filtered demo frames (arm masked via FK silhouette; pre-grasp frames only near the puck); optional scan sweep for future sessions | RoboSplat's released pipeline (Apache-2.0), gsplat as fallback | Medium; task-directed coverage + webcam-grade capture is unproven; test early. |
| 3 | Object Gaussian segmentation + per-episode puck pose | RoboSplat tooling; blob-detect for init | Low-medium. |
| 4 | Per-frame composite render at novel views: static splat + FK-posed arm + phase-correct puck | diff-gaussian-rasterization; URDF-driven arm | Medium; arm compositing seams; z-buffer against splat depth. |
| 5 | Inpaint residual holes (regions no camera ever saw) | cv2.inpaint → LaMa if needed | Low. |
| 6 | Strategy selection over generated variants | See §6, **not** MCTS + validation loss as originally drafted | High as originally designed; see revised design. |

## 5. Revised phased roadmap

- **Phase 0 (spike, ~days, laptop-only).** Entirely on existing `pick_place_v2`
  footage: COLMAP-bootstrap the hand-eye calibration (§3b), select
  low-velocity/masked frames, run RoboSplat's released reconstruction. No physical
  rig access needed. The true unknown: is task-directed webcam footage sufficient
  for a usable splat near the original viewpoints?
- **Phase 1 (~1-2 weeks).** Novel-view episode re-rendering with hand-picked
  camera offsets: static splat + FK arm + puck attachment, composited per frame,
  exported as LeRobot-format synthetic episodes. Train ACT on real+synthetic and
  compare against the PR #3/#5 pixel-augmentation baseline.
- **Phase 2 (~weeks 3-4).** Object-pose augmentation (move the puck's pre-grasp
  pose and re-render the reach phase accordingly, RoboSplat's
  equivariant-transform trick), appearance/lighting edits in splat space,
  inpainting cleanup.
- **Phase 3 (~weeks 5+).** Strategy selection per §6, only if Phase 1-2 show the
  generated data actually moves policy success.

## 6. The search component, honestly reassessed

The original draft proposed MCTS over augmentation configurations, rewarded by
policy validation loss. Two published findings break that design:

1. **Validation loss is a bad reward signal for policies.** The robomimic study
   ("What Matters in Learning from Offline Human Demonstrations," CoRL 2021) found
   the best-validation-loss checkpoint is routinely **50-100% worse in task success**
   than the best checkpoint; surrogate imitation losses simply don't rank policies
   by rollout success. An MCTS loop rewarded by validation loss would efficiently
   optimize the wrong objective.
2. **Expensive augmentation search has a poor track record.** AutoAugment's learned
   search (~15k GPU-hours) was matched or beaten by RandAugment, which is random
   sampling from a sane augmentation space with 2 tunable scalars. The field's
   lesson: invest in the augmentation *space*, not the search *policy*.

Revised design, in order of preference:

- **Default: RandAugment-style random sampling** over validated parameter ranges
  (viewpoint offset bounds, lighting perturbation bounds, object-pose region), with
  the ranges themselves validated once by ablation. This is cheap, matches the
  published evidence, and composes with the existing per-frame pipeline's
  `probability` mechanism.
- **If selection is genuinely needed:** frame it as a low-dimensional
  hyperparameter search (successive halving / Bayesian optimization over ~4-6
  scalars), scored by **small-budget real rollouts** (e.g. 10 rollouts per
  candidate on the actual arm) or, second-best, splat-rendered visual evaluation à
  la RoboGSim, and never by validation loss alone.
- **MCTS specifically** is only defensible if the search space is genuinely
  sequential/compositional (ordered chains of scene edits where order matters and
  the tree structure prunes meaningfully). That's worth one honest experiment as a
  research question ("does searched augmentation beat random augmentation at equal
  compute?"), with RandAugment-style random as the null hypothesis it must beat.

## 7. Where the actual novelty is (narrowed)

The original draft claimed "nobody has combined splat augmentation with search" as
the headline contribution. After the literature check that claim needs shrinking:
RoboSplat already covers the generation side comprehensively, and the
augmentation-search idea carries a negative prior from the AutoAugment→RandAugment
history. What's honestly left:

- **Replication on commodity hardware**: RoboSplat-class results on a ~$100 SO101 +
  webcams + LeRobot stack, integrated into an open per-frame augmentation library,
  is a real engineering contribution even with zero algorithmic novelty.
- **Success-correlated strategy selection** remains genuinely open: *if* one can
  build a cheap reward signal that actually tracks rollout success (which
  validation loss does not), search over scene edits becomes meaningful. That
  reward-signal problem, not the tree search, is the research-shaped hole.

## 8. Integration map: how this bolts onto the existing program

### 8a. The missing keystone: a dataset I/O layer

Nothing in `src/lmfao/` today reads or writes episodes; the core is numpy-only
(`Augmenter` consumes in-memory `(F, H, W, C)` arrays; today's real-data validation
was ad-hoc ffmpeg/PIL scripting outside the package). Both halves of the program
need the same missing piece, **`lmfao.datasets`**: load a LeRobot episode (video
shards + timestamp slicing + state parquet) into arrays, and write an
episode back out in the same format. The v1 pipeline needs it to offer an
end-to-end "augment this HF dataset" entry point at all; miniworld needs it to
emit synthetic episodes. Build it once, before any splat work; it is the first
integration artifact and it de-risks nothing-to-do-with-3D.

### 8b. Division of labor rule (prevents overlap)

Splat-space lighting edits (RoboSplat-style attribute perturbation) and our
`lighting.*` modules do overlapping jobs. To keep the systems from silently
double-applying the same perturbation class:

> **miniworld owns geometry** (camera viewpoint, object pose, embodiment/appearance
> swaps: the things requiring 3D consistency). **The pipeline owns photometrics**
> (lighting, noise, occlusion overlays, crop: the things that are per-frame 2D).

Splat-space lighting stays out of scope unless Phase 2 ablations show pixel-space
lighting is insufficient. One augmentation class, one owner.

### 8c. Ordering convention

Pipeline steps execute sequentially, so config order is semantics. Miniworld is
not a pipeline step; it's the optional parallel source (§1) whose output enters
the same queue as real episodes. The canonical order below applies identically to
every episode, real or synthetic:

```
episode source: real recording  OR  [miniworld render (switchable, geometry)]
  -> occlusion (Ryan: synthetic occluders drawn over the rendered frame)
  -> lighting.* (photometric)
  -> noise (Andrew: sensor-level, near-last)
  -> spatial.random_crop (per-frame, last)
```

Occluders must land *after* rendering (they sit on top of whatever the camera,
real or virtual, sees); noise and crop are sensor-plane effects and go last.
Note the vocabulary hazard for the team sync: Ryan's occlusion module *adds*
occluders; miniworld's inpainting *removes* reconstruction holes. Same word,
opposite operations, zero shared code.

### 8d. Provenance metadata

Synthetic episodes must be distinguishable from real ones downstream. Mirror the
existing `metadata["augmentation_params"]` convention: miniworld stamps each
emitted episode with `{"miniworld": {"source_episode": ..., "camera_offset": ...,
"object_pose_edit": ..., "inpainted_fraction": ...}}`. The pipeline already
accumulates its own params on top; one metadata dict tells the full story of any
training frame.

### 8e. Packaging

Core `lmfao` currently depends on numpy alone; keep it that way. Miniworld's
stack (torch, diff-gaussian-rasterization, RoboSplat) ships as an optional extra
(`pip install lmfao[miniworld]`), so Andrew's and Ryan's modules never inherit a
CUDA dependency. "Lightweight" stays true for the 90% use case.

### 8f. The search/MCTS linkage: one config, both systems

This is the load-bearing integration point. Ryan's pipeline is already
config-driven: `AugmentationPipeline.from_config` takes a list of
`{"name", "params", "probability", "enabled"}` dicts. That schema **is** the
search layer's action space, extended with a miniworld block:

```python
candidate = {
    "miniworld": {"enabled": True, "camera_offsets": [...], "object_pose_region": [...], "n_synthetic": 40},
    "pipeline":  [
        {"name": "lighting.color_temperature", "params": {"intensity": 0.35}, "probability": 0.5},
        {"name": "spatial.random_crop", "params": {"pad": 8}, "probability": 1.0},
    ],
}
```

The strategy selector (§6: random sampling by default, BO/successive-halving or
MCTS if justified) proposes candidates in exactly this shape; evaluation is
`candidate -> miniworld render -> pipeline -> proxy train -> success-correlated
score`; the winner ships as a plain JSON config the team can commit. The
`"enabled"` flag mirrors the pipeline modules' own `enabled` convention: the
search layer (or a human) can switch the synthetic branch off entirely and the
config degenerates to a plain v1 pipeline config. No new
plumbing is needed on the pipeline side; the search layer is a *client* of the
existing config interface, which also means it can tune Andrew's and Ryan's module
parameters the day they land, with zero changes to their code. MCTS's tree, if
used, branches over **ordered edit sequences** (per §8c, order is semantics, so
the sequential structure MCTS needs actually exists here).

### 8g. Product surface: every user option is either Adjust or Generate

From the user's perspective the whole system exposes exactly two kinds of
options, and every knob belongs to one of them:

- **Adjust**: reshape episodes that already exist (the `"pipeline"` list:
  `lighting.*`, occlusion, noise, `spatial.random_crop`). Pixel-space, CPU-cheap,
  always available. Changes how frames *look*; episode count is unchanged.
- **Generate**: use the splat to manufacture episodes that were never recorded
  (the `"miniworld"` block: `enabled`, `n_synthetic`, `camera_offsets`,
  `object_pose_region`). Scene-space, GPU, requires the one-time reconstruction.
  Changes what *exists*; you end up with more episodes than you recorded, and
  Adjust then seasons all of them identically.

This is the same split as §8b's division-of-labor rule, stated as UX instead of
architecture: Adjust = photometrics, Generate = geometry. Docs, config schema,
and any future CLI should use these two words consistently so a user never has
to wonder which layer a knob lives in.

## 9. Open decisions

1. Ship this doc as the v2 plan (today's "slide"), no code.
2. Scaffold `lmfao/miniworld/` interface stubs (`SceneReconstructor`,
   `BeaconTracker`, `NovelViewRenderer`, `Inpainter`, `StrategySelector`); cheap,
   no heavy deps.
3. Build `lmfao.datasets` (§8a); useful to v1 immediately, prerequisite for v2.
4. Run Phase 0 for real: COLMAP-bootstrap calibration + splat fit from existing
   `pick_place_v2` footage (§3b). Laptop-only; no physical rig access required,
   so it can start any time.
