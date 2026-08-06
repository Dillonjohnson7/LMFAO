#!/bin/bash
# ============================================================================
# 150k-run watchdog. Runs on the workcell. Every 10 min:
#   - rsync train.log + gpu.log + checkpoints off the pod  -> ./run150k/
#   - regenerate loss_curve.png
#   - write status.txt (step, loss, steps/s, spend)
# Exits (thereby notifying Claude) ONLY on: DONE / SSH-DEAD / STALL / COST-CAP.
# On DONE: verifies final sync, uploads last model to the Hub, TERMINATES the pod.
#
# ARCHIVE / NON-RERUNNABLE: a record of the completed 150k run. It reads a
# baseline_balance.txt from an ephemeral /tmp scratchpad now gone, and its pod is
# terminated, so it will not run as-is. The reusable RunPod driver was preserved
# at training/rp.py; point future runs there.
# ============================================================================
set -u
D=$HOME/SO101_policy/training/run150k
SM=/tmp/scratchpad
read IP PORT < "$D/pod_ssh.txt"
POD=$(cat "$D/pod_id.txt")
B0=$(cat "$SM/baseline_balance.txt")
VENV=$HOME/SO101_policy/.venv
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=10 -o BatchMode=yes -i $HOME/.ssh/id_ed25519"

ssh_fail=0; last_step=-1; stall=0; cost_warned=0

balance() { cd "$SM" && python3 rp.py list 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['myself']['clientBalance'])" 2>/dev/null; }

while true; do
  NOW=$(date '+%F %T')
  # --- sync ---
  if rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" \
       root@$IP:/root/train.log root@$IP:/root/gpu.log root@$IP:/root/train.done "$D/" 2>/dev/null; then
    ssh_fail=0
  else
    # train.done may not exist yet — retry logs alone before counting a failure
    if rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" root@$IP:/root/train.log "$D/" 2>/dev/null; then
      ssh_fail=0
    else
      ssh_fail=$((ssh_fail+1))
      echo "$NOW ssh_fail=$ssh_fail" >> "$D/watchdog.log"
      if [ $ssh_fail -ge 3 ]; then echo "ALERT-SSH: pod unreachable 3 cycles" | tee -a "$D/watchdog.log"; exit 2; fi
      sleep 600; continue
    fi
  fi
  rsync -az --timeout=300 -e "ssh $SSH_OPTS -p $PORT" \
    root@$IP:/root/outputs/train/act_pick_place/checkpoints "$D/" 2>/dev/null

  # --- curve + status ---
  "$VENV/bin/python" "$D/plot_loss.py" > "$D/last_plot.txt" 2>&1
  STEP=$(grep -aoE '[0-9]+/150000' "$D/train.log" 2>/dev/null | tail -1 | cut -d/ -f1); STEP=${STEP:-0}
  LOSS=$(grep -aoE 'loss:[0-9.]+' "$D/train.log" 2>/dev/null | tail -1 | cut -d: -f2); LOSS=${LOSS:-na}
  BAL=$(balance); SPENT=$(python3 -c "print(f'{$B0-${BAL:-$B0}:.2f}')" 2>/dev/null || echo "?")
  echo "$NOW step=$STEP loss=$LOSS balance=$BAL spent_this_run=\$$SPENT" | tee "$D/status.txt" >> "$D/watchdog.log"

  # --- done? ---
  if [ -s "$D/train.done" ]; then
    echo "TRAINING DONE ($(cat "$D/train.done")) — final sync + upload + terminate" | tee -a "$D/watchdog.log"
    rsync -az --timeout=600 -e "ssh $SSH_OPTS -p $PORT" \
      root@$IP:/root/outputs/train/act_pick_place/checkpoints "$D/" 2>/dev/null
    rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" root@$IP:/root/train.log root@$IP:/root/gpu.log "$D/" 2>/dev/null
    if [ -f "$D/checkpoints/last/pretrained_model/model.safetensors" ]; then
      "$VENV/bin/hf" upload Dillonjohnson/act_pick_place "$D/checkpoints/last/pretrained_model" ckpt-150k-last \
        --repo-type=model >> "$D/watchdog.log" 2>&1 && echo "HUB UPLOAD OK" >> "$D/watchdog.log"
    fi
    cd "$SM" && python3 rp.py terminate "$POD" >> "$D/watchdog.log" 2>&1
    echo "POD TERMINATED. Run complete." | tee -a "$D/watchdog.log"
    exit 0
  fi

  # --- stall? ---
  if [ "$STEP" = "$last_step" ] && [ "$STEP" != "0" ]; then
    stall=$((stall+1))
    if [ $stall -ge 2 ]; then echo "ALERT-STALL: step $STEP unchanged 2 cycles" | tee -a "$D/watchdog.log"; exit 4; fi
  else
    stall=0
  fi
  last_step=$STEP

  # --- cost caps ---
  if [ -n "${BAL:-}" ]; then
    OVER6=$(python3 -c "print(1 if $B0-$BAL>6.0 else 0)")
    OVER75=$(python3 -c "print(1 if $B0-$BAL>7.5 else 0)")
    if [ "$OVER75" = "1" ]; then
      echo "COST HARD-CAP (\$7.50) — terminating pod" | tee -a "$D/watchdog.log"
      cd "$SM" && python3 rp.py terminate "$POD" >> "$D/watchdog.log" 2>&1
      exit 3
    fi
    if [ "$OVER6" = "1" ] && [ $cost_warned = 0 ]; then
      echo "COST ALERT: >\$6 spent this run" >> "$D/watchdog.log"; cost_warned=1
    fi
  fi

  sleep 600
done
