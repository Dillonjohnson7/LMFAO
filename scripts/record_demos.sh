#!/usr/bin/env bash
# Recollect fresh SO101 pick-and-place demonstrations via the guided recorder
# (so101/rec — ENTER / keep / redo, with episode counter).
#
# Must run on the robot workstation (leader + follower + cams).
#
# Usage:
#   ./scripts/record_demos.sh 40
#   TASK="pick the blue puck and place it in the brown box" ./scripts/record_demos.sh
#   DATASET_ROOT=/path/to/pick_place_v3 LEROBOT_VENV=/home/multiply/envs/lerobot06 \
#     ./scripts/record_demos.sh 40
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="$ROOT/so101/rec"

if [[ ! -x "$REC" ]]; then
  echo "missing guided recorder: $REC" >&2
  exit 1
fi

echo "Recording via guided SO101 recorder (so101/rec)."
echo "  ENTER advances · keep/redo at STEP 3 · empty episodes are never saved."
echo "  Pin cams first if needed: python so101/eval/cam_check.py --find-front|--find"
echo

exec "$REC" "$@"
