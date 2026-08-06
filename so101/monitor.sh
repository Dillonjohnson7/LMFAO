#!/bin/bash
# Single-token launcher for the follower USB monitor, uv-managed env (replaces conda).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
exec python "$SCRIPT_DIR/monitor_follower.py" "$@"
