# LMFAO — State of the Art & Next Steps

Living status doc. Last updated 2026-08-05. This is the single source of truth for
where the project is, what's proven, what we learned, and what to do next.

LMFAO is a **command-line tool for augmenting robot-learning datasets**: point it
at a [LeRobot](https://github.com/huggingface/lerobot) v3 dataset (local folder or
a Hugging Face link) and it turns each recorded episode into many varied training
clips, written back out as a ready-to-train LeRobot dataset.

> ## ⚠️ SECURITY — READ BEFORE YOU COMMIT (humans and AI agents)
>
> **This is a PUBLIC GitHub repo.** A real personal SSH private key was once
> committed here via `git add -A` (see §3). Whoever works on this repo next — a
> person or an AI agent in Cursor/Claude/etc. — MUST follow these rules:
>
> 1. **NEVER `git add -A`, `git add .`, or `git add -u`.** Stage explicit paths
>    only: `git add src/lmfao/... tests/... docs/STATUS.md`. Run `git status` and
>    read it before every commit.
> 2. **NEVER write secrets, keys, tokens, `.env`, or credentials inside the repo
>    tree** — not even temporarily. Use a scratch dir OUTSIDE the repo (e.g. `/tmp`).
>    Never run `ssh-keygen -f <path-in-repo>`.
> 3. **Before any commit, run the secret scan** (a `gitleaks` hook is wired up):
>    `git diff --cached` should contain zero key material; if gitleaks flags
>    anything, STOP.
> 4. **Never paste API keys / tokens into the chat or terminal transcript.** If a
>    credential is needed, have the human export it as an env var themselves.
> 5. **If a secret ever reaches a commit:** treat it as compromised, purge it from
>    history, force-push, AND have the owner rotate/revoke it (GitHub caches
>    unreachable blobs; public repos get scraped and forked within minutes).
>
> Full incident writeup and the current guardrails are in §3.

---

## 1. Current state of the art (what works, verified)

### The CLI (`lmfao`)
Four entry points, all native to the LeRobot v3 on-disk format:

| command | status | what it does |
| --- | --- | --- |
| `lmfao` (no args) | ✅ working | interactive wizard: paste an HF link, pick effects, run |
| `lmfao augment` | ✅ **production-ready** | ADJUST pipeline at native resolution — the trainable path |
| `lmfao inspect` | ✅ working | summarize any LeRobot dataset without decoding video |
| `lmfao generate` | ⚠️ experimental | synthesize novel camera views (low-res reference renderer) |

`augment` is the mature command: full 6-effect pipeline (lighting ×3, noise ×2,
occlusion, spatial crop), deterministic **magnitude sweeps** (e.g. brightness
±5/±10/±15% with a projected video count), `--variants` N seasoned copies,
`--include-original`, a **streaming one-episode-per-file writer** (bounded memory),
and crash-safe **`--resume`** (checkpoint per source episode; verified bit-identical
to a clean run). Speed: LUT lighting fast path (~32× faster brightness, memory
19 GB → 2 GB) + `veryfast` x264 preset.

### LeRobot training compatibility — CONFIRMED
A full RunPod smoke test (RTX A4500) **loaded an augmented `pick_place_v2` in
lerobot 0.6.2 and trained an ACT policy on GPU**, loss decreasing 74 → 22 over 10
steps. Getting there required matching the real v3.0 schema exactly (all four now
emitted by both writers):
1. `meta/stats.json` + per-episode `stats/<feat>/<key>` columns (LeRobot normalizes
   from these).
2. `tasks.parquet` with pandas index metadata (task string as the DataFrame index).
3. `info.json` declares **every** data column as a feature — state, action, video,
   **and** timestamp/frame_index/episode_index/index/task_index — plus `splits` and
   the size fields.
4. state/action stored as `fixed_size_list<float32>[dim]`, not variable `list<>`.

### Hardening & quality
- **~76 bugs fixed** across three adversarial multi-agent sweeps (data-loss paths,
  silent corruption, format bugs, crashes → clean errors).
- **209 tests, ruff clean.** Reader/writer round-trips, resume correctness,
  streaming, LUT bit-identity, wizard flows, CLI error paths.

### Branch reconciliation (done)
`feature/lerobot-export` and `feature/lerobot-web-export` are **retired** (deleted
local + origin, archived as tags `archive/lerobot-export`,
`archive/lerobot-web-export`). Their unique value was ported onto main: LUT
lighting, streaming writer + `--resume`, and the browser LeRobot-folder ingest +
job.json export in `web/`.

### Canonical data
- **Dev / smoke dataset:** HF `Dillonjohnson/pick_place_v2` (SO101 / `so_follower`,
  wrist camera) — used to build and prove LMFAO's LeRobot export + ACT load.
- **Eval dataset:** a **newly recorded** session (see `so101/rec` / `scripts/record_demos.sh` /
  `docs/TRAINING_RUN.md`). Do not treat `pick_place_v2` as the comparative-eval
  training set. Never the old Downloads teleop folder.

---

## 2. Key technical learnings

**LeRobot v3.0 dataset format (the exact requirements):** see §1's four points.
LeRobot derives the parquet schema from `info.json` features and passes it to
`datasets.Dataset.from_parquet`, so an undeclared column or a variable-length list
fails the load. Verified against the real `pick_place_v2` (the ground-truth dataset
that loads) and lerobot 0.6.1/0.6.2 loader source.

**Training environment recipe (RunPod, reproduces the SO101 `setup.sh`):**
lerobot 0.6.x is **git-only and needs Python 3.12** (PyPI `lerobot` is 0.4.4 and
can't read v3.0). Fast path on a PyTorch pod image:
```
uv venv --python 3.12 /workspace/v312
uv pip install --python /workspace/v312 \
  "git+https://github.com/huggingface/lerobot.git" datasets "av>=15.0.0,<16.0.0" torchcodec accelerate
```
Train (ACT smoke):
```
lerobot-train --dataset.repo_id=X --dataset.root=/path \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=N --batch_size=2 --save_checkpoint=false --wandb.enable=false
```
`torchcodec` fails to load `libnvrtc` on the plain image and falls back to `pyav`
(harmless). `av` MUST be pinned to 15.x — 18.0 removed `av.option` and breaks lerobot.

**macOS `tar` AppleDouble gotcha:** shipping a dataset to a Linux pod via macOS
`tar` creates `._*.parquet` sidecar files that lerobot's glob reads as bogus
parquets ("magic bytes not found in footer"). This masqueraded as a format bug for
a while. Fix: `COPYFILE_DISABLE=1 tar ...`, or `huggingface-cli upload`, or
`find <dir> -name '._*' -delete` on the pod.

**Other environment facts:** pyarrow `read_table` deadlocks its thread pool on
pyarrow 25 / CPython 3.14 → the reader uses `use_threads=False`. `torch` cannot be
installed into the LMFAO venv — it deadlocks `import av` (so GPU-accelerated noise
stays out of the dataset-touching env). The repo lives under `~/Desktop` which is
**iCloud-synced**; writing datasets triggers sync storms that spike system load and
make commands time out — write large outputs to a non-synced dir like
`~/lmfao_augmented/`.

---

## 3. Security learnings (a real incident — do not repeat)

**What happened:** a real personal SSH private key sitting in the repo root (a
garbled escape-character filename) was committed to the **public** GitHub repo via
`git add -A`, which blindly stages every untracked file.

**Remediation done:** removed from HEAD, purged from all git history
(`filter-branch` + dropped backup refs + gc), force-pushed; added `.gitignore`
guards (`*.pub`, `*.pem`, `*_key`, `id_ed25519*`, `id_rsa*`, `pod_key*`,
`.runpod_key`); scanned every tracked file (no other secrets); the RunPod API key
was never in the repo. The user **revoked the RunPod API key** and **rotated the
SSH key** (old backed up at `~/.ssh/id_ed25519.OLD-COMPROMISED`).

**Hard rules going forward (non-negotiable)** — see the banner at the top of this
doc for the full list. In short: no `git add -A`/`.`/`-u` (explicit paths only,
review `git status` first); never write secrets inside the repo tree; run the
gitleaks scan before committing; never paste keys into the transcript; treat any
leaked secret as compromised and rotate/revoke it.

**Current guardrails in place:**
- `.gitignore` blocks key material: `*.pub`, `*.pem`, `*_key`, `id_ed25519*`,
  `id_rsa*`, `pod_key*`, `.runpod_key`.
- A **gitleaks** secret-scanner runs on commit ("no leaks found" required to pass).
  Verify it's active with `git config --get core.hooksPath` / check for a
  `pre-commit` hook; if missing on a fresh clone, re-enable it before committing.

**Verified clean state (2026-08-04):** key purged from all history + force-pushed;
every tracked file scanned — no other secrets; RunPod API key was never in the repo
and has been revoked; SSH key rotated (old at `~/.ssh/id_ed25519.OLD-COMPROMISED`,
delete after finishing the swap on any servers).

---

## 4. Next steps (prioritized)

**A. Security — finish and verify (highest priority)**
- Finish the SSH rotation: add the new public key to any remote servers'
  `authorized_keys`, test login, remove the old key everywhere, then delete
  `~/.ssh/id_ed25519.OLD-COMPROMISED{,.pub}`.
- Confirm the gitleaks pre-commit hook is active on this clone (and set it up on any
  new clone before the first commit).
- Audit the OTHER repos under `~/Desktop/LocalProjects` for the same mistake — the
  stray key file suggests `git add -A` may have leaked elsewhere too. Quick check
  per repo: `git ls-files | grep -iE 'key|\.pub$|\.pem$|\.env'` and
  `git log --all --diff-filter=A --name-only | grep -iE 'key|\.pub$'`.
- Consider enabling GitHub **push protection / secret scanning** on the repo
  (Settings → Code security) so GitHub itself blocks future secret pushes.

**B. Comparative eval run (the real goal)** — scaffolding landed
Everything is proven at smoke scale. The next experiment is **not** reusing
`pick_place_v2` — it is:

1. **Recollect** fresh SO101 demos (`so101/rec` / `scripts/record_demos.sh` on the robot box)
2. **Augment** with the CLI into matched buckets (stock / lighting / noise /
   occlusion / spatial / full) via `scripts/augment_eval_buckets.sh`
3. **Train** one ACT policy per bucket with identical hparams
   (`scripts/train_eval_buckets.sh` on a CUDA pod)
4. **Compare** physical rollout success under held-out lighting / framing /
   occlusion — that is how we tell whether the CLI helps

Checklist + table: `docs/TRAINING_RUN.md`. Bucket configs:
`configs/training/buckets/*.json`. Transfer with `COPYFILE_DISABLE=1 tar` or
`huggingface-cli upload` (avoid AppleDouble). This cloud agent has **no robot
and no GPU** — recording is on the SO101 workstation; training on RunPod.

**C. `lmfao generate` quality (experimental → trainable)**
The novel-view path produces 96×54, blurry, assumed-pose frames. Making it
trainable needs the real Gaussian-splat backend + FK-derived camera poses — the
multi-week Phase 0-1 in `v2_mini_world_generator_plan.md`. Genuine research risk
(is task-directed webcam footage good enough for a usable splat?).

**D. Nice-to-haves**
- Carry the source dataset's per-feature names (joint names) through `augment` so
  the output `info.json` is fully faithful (currently `names: null`).
- Port the archived export branch's `--encoder`/`--encode-bitrate` hardware-encode
  flags (GPU encoding via `h264_videotoolbox`) — preserved in
  `archive/lerobot-export`.
- Move the repo off `~/Desktop` to stop the iCloud sync storms.
- Expose `--resume` (and a `--dry-run` output-count preview) in the wizard, not just
  the flag CLI.

---

## Related docs
- `README.md` — user-facing CLI quickstart.
- `docs/TRAINING_RUN.md` — full-scale augment → ACT checklist and scripts.
- `docs/v2_mini_world_generator_plan.md` — the GENERATE/splat roadmap (design doc,
  not implemented).
- `docs/adding_features.md` — contributor guide for new augmenters.
- `docs/library-usage-vision.md` — the (partly superseded) Python-library vision.
