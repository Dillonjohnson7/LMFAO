#!/usr/bin/env bash
# SO101 leader → follower teleop (no dataset recording).
# On start the follower enables torque and snaps to the leader's pose —
# pre-match the arms and keep hands clear. Ctrl-C to stop.
set -euo pipefail
SO101_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SO101_DIR/env.sh"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/so101_follower}"
LEADER_PORT="${LEADER_PORT:-/dev/so101_leader}"

exec lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=leader
