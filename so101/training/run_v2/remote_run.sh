#!/usr/bin/env bash
# ============================================================================
# POD-SIDE launcher (copied to /root by launch.sh; run AFTER cloud_setup.sh).
# Starts training detached with the exact file contract that watchdog.sh and
# dashboard.py expect — in v1 this was ad-hoc scratchpad commands, now baked:
#   /root/train.log    trainer output, appended live
#   /root/gpu.log      one nvidia-smi sample every 300 s
#   /root/train.done   written ONLY when training exits (holds exit code+time)
# Survives SSH disconnect (nohup + setsid). Idempotence guard: refuses to start
# if a lerobot-train is already running.
# ============================================================================
set -euo pipefail
cd /root

if pgrep -f "bin/lerobot-train" >/dev/null 2>&1; then
  echo "REFUSING: a lerobot-train is already running (pgrep bin/lerobot-train)."
  exit 1
fi
rm -f train.done

setsid nohup bash -c '
  ( while true; do
      echo "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)" >> /root/gpu.log
      sleep 300
    done ) &
  GPU_PID=$!
  bash /root/train_act.sh >> /root/train.log 2>&1
  RC=$?
  kill $GPU_PID 2>/dev/null
  echo "exit=$RC $(date -u +%FT%TZ)" > /root/train.done
' > /root/nohup.out 2>&1 &

sleep 2
echo "training launched — tail -f /root/train.log"
tail -c 400 /root/train.log 2>/dev/null || true
