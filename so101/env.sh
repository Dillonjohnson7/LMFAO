#!/usr/bin/env bash
# Shared env activation for SO101 workcell tools under LMFAO/so101.
# Prefer LEROBOT_VENV, then common NUC path, then a local .venv next to this dir.
SO101_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SO101_ROOT="${SO101_ROOT:-$SO101_DIR}"

_activate() {
  local candidate="$1"
  if [[ -f "$candidate/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$candidate/bin/activate"
    return 0
  fi
  return 1
}

if [[ -n "${LEROBOT_VENV:-}" ]]; then
  _activate "$LEROBOT_VENV" || {
    echo "LEROBOT_VENV=$LEROBOT_VENV has no bin/activate" >&2
    return 1 2>/dev/null || exit 1
  }
elif _activate /home/multiply/envs/lerobot06; then
  :
elif _activate "$SO101_DIR/.venv"; then
  :
elif _activate "$(cd "$SO101_DIR/.." && pwd)/.venv"; then
  :
else
  echo "No LeRobot venv found. Set LEROBOT_VENV=/path/to/venv" >&2
  return 1 2>/dev/null || exit 1
fi
