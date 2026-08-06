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
