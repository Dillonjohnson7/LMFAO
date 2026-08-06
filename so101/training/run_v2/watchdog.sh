#!/bin/bash
# ============================================================================
# V2-run watchdog. Runs on the workcell. Every 10 min:
#   - rsync train.log + gpu.log + train.done + checkpoints off the pod -> run_v2/
#   - write status.txt (step, loss, balance, spend)
# Exits (thereby notifying the operator/agent) ONLY on:
#   0 DONE (after final sync + Hub upload + pod TERMINATE)
#   2 SSH-DEAD (3 cycles)   3 COST-CAP (pod terminated)   4 STALL (2 cycles)
# NOTE on 2/4 the pod is LEFT RUNNING (billing!) for a human decision — check
# rp.py list immediately.
#
# Rebuilt 2026-07-21 from the archived run150k/watchdog.sh: all paths repo-local
# (the v1 copy read baseline/key from a /tmp scratchpad a reboot erased),
# job = act_pick_place_v2, upload dir = ckpt-v2-last, cost caps sized to the
# $6.32 balance (alert $4 / hard-kill $5 this run; est. run $2.20-3.30).
# Prereqs: launch.sh wrote pod_ssh.txt, pod_id.txt, baseline_balance.txt here.
# ============================================================================
set -u
D=$HOME/SO101_policy/training/run_v2
RP=$HOME/SO101_policy/training/rp.py
VENV=$HOME/SO101_policy/.venv
JOB="${JOB:-act_pick_place_v2}"
TOTAL="${TOTAL:-120000}"
read IP PORT < "$D/pod_ssh.txt"
POD=$(cat "$D/pod_id.txt")
B0=$(cat "$D/baseline_balance.txt")
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=10 -o BatchMode=yes -i $HOME/.ssh/id_ed25519"

ssh_fail=0; last_step=-1; stall=0; cost_warned=0

balance() {
  "$VENV/bin/python" "$RP" list 2>/dev/null \
    | "$VENV/bin/python" -c "import sys,json;print(json.load(sys.stdin)['data']['myself']['clientBalance'])" 2>/dev/null
}

while true; do
  NOW=$(date '+%F %T')
  # --- sync ---
  if rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" \
       root@$IP:/root/train.log root@$IP:/root/gpu.log root@$IP:/root/train.done "$D/" 2>/dev/null; then
    ssh_fail=0
  else
    # train.done may not exist yet — retry logs alone before counting a failure.
    # On failure KEEP THE STDERR (Mistake #6: a hidden error cost the first run
    # 40 min of blind ssh_fail cycles — the actual cause was rsync missing
    # remotely, visible instantly once stderr was read).
    if rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" root@$IP:/root/train.log "$D/" 2>>"$D/watchdog.log"; then
      ssh_fail=0
    else
      ssh_fail=$((ssh_fail+1))
      echo "$NOW ssh_fail=$ssh_fail" >> "$D/watchdog.log"
      if [ $ssh_fail -ge 3 ]; then echo "ALERT-SSH: pod unreachable 3 cycles — POD STILL BILLING, check rp.py list" | tee -a "$D/watchdog.log"; exit 2; fi
      sleep 600; continue
    fi
  fi
  rsync -az --timeout=300 -e "ssh $SSH_OPTS -p $PORT" \
    root@$IP:/root/outputs/train/$JOB/checkpoints "$D/" 2>/dev/null

  # --- status (TOTAL auto-detected from the log's own tqdm bar — works for any
  # STEPS override; the env default is only a pre-log fallback) ---
  LOGTOTAL=$(grep -aoE '[0-9]+/[0-9]+ \[' "$D/train.log" 2>/dev/null | tail -1 | sed -E 's|[0-9]+/([0-9]+) \[|\1|')
  TOTAL=${LOGTOTAL:-$TOTAL}
  STEP=$(grep -aoE "[0-9]+/$TOTAL" "$D/train.log" 2>/dev/null | tail -1 | cut -d/ -f1); STEP=${STEP:-0}
  LOSS=$(grep -aoE 'loss:[0-9.]+' "$D/train.log" 2>/dev/null | tail -1 | cut -d: -f2); LOSS=${LOSS:-na}
  BAL=$(balance); SPENT=$("$VENV/bin/python" -c "print(f'{$B0-${BAL:-$B0}:.2f}')" 2>/dev/null || echo "?")
  echo "$NOW step=$STEP loss=$LOSS balance=$BAL spent_this_run=\$$SPENT" | tee "$D/status.txt" >> "$D/watchdog.log"

  # --- done? ---
  if [ -s "$D/train.done" ]; then
    echo "TRAINING DONE ($(cat "$D/train.done")) — final sync + upload + terminate" | tee -a "$D/watchdog.log"
    rsync -az --timeout=600 -e "ssh $SSH_OPTS -p $PORT" \
      root@$IP:/root/outputs/train/$JOB/checkpoints "$D/" 2>/dev/null
    rsync -az --timeout=60 -e "ssh $SSH_OPTS -p $PORT" root@$IP:/root/train.log root@$IP:/root/gpu.log "$D/" 2>/dev/null
    if [ -f "$D/checkpoints/last/pretrained_model/model.safetensors" ]; then
      "$VENV/bin/hf" upload Dillonjohnson/act_pick_place "$D/checkpoints/last/pretrained_model" ckpt-v2-last \
        --repo-type=model >> "$D/watchdog.log" 2>&1 && echo "HUB UPLOAD OK (ckpt-v2-last)" >> "$D/watchdog.log"
    else
      echo "WARN: no model.safetensors in final sync — DO NOT TERMINATE by hand until saved" | tee -a "$D/watchdog.log"
    fi
    "$VENV/bin/python" "$RP" terminate "$POD" >> "$D/watchdog.log" 2>&1
    echo "POD TERMINATED. Run complete." | tee -a "$D/watchdog.log"
    exit 0
  fi

  # --- stall? ---
  if [ "$STEP" = "$last_step" ] && [ "$STEP" != "0" ]; then
    stall=$((stall+1))
    if [ $stall -ge 2 ]; then echo "ALERT-STALL: step $STEP unchanged 2 cycles — POD STILL BILLING, check rp.py list" | tee -a "$D/watchdog.log"; exit 4; fi
  else
    stall=0
  fi
  last_step=$STEP

  # --- cost caps (baseline $6.32; 150k run estimate $3.75-4.20; USER DECISION
  # 07-21: kill raised $5.00 -> $5.75 to guarantee 150k on a slow box) ---
  if [ -n "${BAL:-}" ]; then
    OVER_ALERT=$("$VENV/bin/python" -c "print(1 if $B0-$BAL>4.5 else 0)")
    OVER_KILL=$("$VENV/bin/python" -c "print(1 if $B0-$BAL>5.75 else 0)")
    if [ "$OVER_KILL" = "1" ]; then
      echo "COST HARD-CAP (\$5.75 this run) — saving mirror ckpt, then terminating pod" | tee -a "$D/watchdog.log"
      # the mirror is at most one 10-min cycle stale (~2k steps) — save it before killing
      if [ -f "$D/checkpoints/last/pretrained_model/model.safetensors" ]; then
        "$VENV/bin/hf" upload Dillonjohnson/act_pick_place "$D/checkpoints/last/pretrained_model" ckpt-v2-costcap \
          --repo-type=model >> "$D/watchdog.log" 2>&1 && echo "HUB UPLOAD OK (ckpt-v2-costcap)" >> "$D/watchdog.log"
      fi
      "$VENV/bin/python" "$RP" terminate "$POD" >> "$D/watchdog.log" 2>&1
      exit 3
    fi
    if [ "$OVER_ALERT" = "1" ] && [ $cost_warned = 0 ]; then
      echo "COST ALERT: >\$4.50 spent this run" >> "$D/watchdog.log"; cost_warned=1
    fi
  fi

  sleep 600
done
