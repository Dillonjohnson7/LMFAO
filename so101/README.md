# SO101 workcell — teleop & data collection

Brought in from [MultiplyLabor/SO101_policy](https://github.com/MultiplyLabor/SO101_policy)
so LMFAO owns the full loop: **record → augment → train → compare**.

## Daily drivers

| command | what |
| --- | --- |
| `./so101/setup.sh` | one-time udev names + install calib JSONs |
| `python so101/eval/cam_check.py --find-front` / `--find` | pin front / wrist cams |
| `./so101/teleop.sh` | leader → follower teleop (no dataset) |
| `./so101/rec [n]` | **guided** demo recorder (ENTER / keep / redo) |
| `python so101/view_episode.py [ep]` | side-by-side H.264 review of a saved episode |
| `python so101/relax.py` | limp the follower to reposition by hand |

Or from the repo root: `./scripts/record_demos.sh 40` (thin wrapper around `so101/rec`).

## Prior policy work (pulled in from SO101_policy, 2026-08-06)

- `CHECKPOINT.md` — canonical record of every training run and rollout, with
  lessons/mistakes. **Status as of 2026-07-22: V2 WORKS** — wrist-cam retrain
  (45 demos, 2 cams, 150k steps) completes the full pick→carry→place task
  (4/13 on first night). v1 (front-cam-only) never placed once.
- `V2_PIPELINE.md` — the wrist-cam re-record + retrain runbook.
- `NEXT_STEPS.md` — outstanding plan items.
- `STATUS.md` — teleop/USB/calibration foundation notes.
- `tools/creep.sh` — the rollout test harness (wraps `eval/creep_test.py`;
  `check` = safe no-robot selftest, `go`/`go2-*` = live rollout modes).
- `eval/score_run.py`, `eval/offline_eval.py` — rollout scoring / offline eval.
- `training/` — training-run configs, RunPod launch/watchdog scripts, and
  loss-curve charts from the v1 vs v2 runs.

## Guided recorder

```bash
# on the robot box, LeRobot env active (or LEROBOT_VENV set):
./so101/rec 40
```

Flow per episode:

1. **SET UP** — puck in the usual spot, pose the leader  
2. **RECORD** — teleop the pick & place; **ENTER** stops  
3. **KEEP?** — ENTER keep · `r` redo · `q` save & quit  
4. **RESET** — prep the next demo (same puck location each time; no correction demos)  

Empty / faulted episodes are discarded. Sessions resume into the same
`DATASET_ROOT` if `meta/` already exists.

Defaults:

- Dataset: `so101/datasets/pick_place_v3`
- Cams: front + wrist (`WRIST=1`), pinned via `front_cam.path` / `wrist_cam.path`
- Task: `pick the blue puck and place it in the brown box`
- Ports: `/dev/so101_follower` / `/dev/so101_leader`

Override with env: `TASK`, `EPISODES`, `DATASET_ROOT`, `WRIST=0`, `FOLLOWER_PORT`,
`LEADER_PORT`, `LEROBOT_VENV`, `SO101_ROOT`.

## After recording

Point LMFAO at the local dataset root:

```bash
export STOCK="$PWD/so101/datasets/pick_place_v3"
./scripts/augment_eval_buckets.sh
```

See `docs/TRAINING_RUN.md` for the full recollect → augment → train → compare plan.

## Safety

On connect the follower **energizes and snaps to the leader**. Match poses by
hand first and keep clear. Prefer direct USB (no hub) for the arms.
