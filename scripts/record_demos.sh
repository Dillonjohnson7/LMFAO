#!/usr/bin/env bash
# Recollect fresh SO101 pick-and-place demonstrations with LeRobot.
#
# This must run on the robot workstation (leader + follower plugged in).
# LMFAO does not record teleop — it only augments what you collect here.
#
# Prereqs (robot machine):
#   - LeRobot 0.6.x with so101 support
#   - Arms calibrated: lerobot-calibrate ...
#   - hf auth login   (if pushing the dataset)
#
# Usage:
#   FOLLOWER_PORT=/dev/ttyACM0 LEADER_PORT=/dev/ttyACM1 \
#   HF_USER=Dillonjohnson TASK="pick the cube and place it in the bin" \
#   ./scripts/record_demos.sh
set -euo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:?set FOLLOWER_PORT (e.g. /dev/ttyACM0)}"
LEADER_PORT="${LEADER_PORT:?set LEADER_PORT (e.g. /dev/ttyACM1)}"
HF_USER="${HF_USER:?set HF_USER (Hugging Face username)}"
DATASET_NAME="${DATASET_NAME:-pick_place_v3}"
REPO_ID="${REPO_ID:-${HF_USER}/${DATASET_NAME}}"
NUM_EPISODES="${NUM_EPISODES:-50}"
TASK="${TASK:-pick the cube and place it in the bin}"
FOLLOWER_ID="${FOLLOWER_ID:-so101_follower}"
LEADER_ID="${LEADER_ID:-so101_leader}"
# Match the camera setup you will train and evaluate with. Wrist-only matches
# the prior pick_place_v2 convention; add a front cam if you want both.
CAMERAS="${CAMERAS:-{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

echo "Recording ${NUM_EPISODES} episodes -> ${REPO_ID}"
echo "Task: ${TASK}"
echo
echo "Protocol tips:"
echo "  - Keep lighting / camera / table layout fixed during the whole session."
echo "  - Prefer slow, consistent teleop; redo bad episodes (←)."
echo "  - Optional hygiene: after task episodes, record a ~30s slow scene sweep"
echo "    (helps future GENERATE / splat work; not required for ADJUST)."
echo "  - Keys: → next episode, ← redo, ESC finish."
echo

exec lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK" \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --display_data=true
