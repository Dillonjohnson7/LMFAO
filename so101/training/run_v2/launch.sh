#!/bin/bash
# ============================================================================
# ONE-COMMAND v2 run launcher (workcell side).      bash training/run_v2/launch.sh
#
# deploy A5000 -> wait for SSH -> write pod files -> copy scripts -> cloud_setup
# (with local HF token) -> start training (remote_run.sh) -> start watchdog +
# dashboard locally. Every stage gated; on failure AFTER deploy it stops and
# prints the pod id — the pod is BILLING until you rp.py terminate it.
#
# Run AFTER (V2_PIPELINE.md §3-4): push_dataset.py + a passing smoke_test.sh.
# ============================================================================
set -euo pipefail
D=$HOME/SO101_policy/training/run_v2
T=$HOME/SO101_policy/training
VENV=$HOME/SO101_policy/.venv
# 120k, not 150k (runbook decision): v1's tail was measured flat past ~100k, and
# 120k preserves budget for a contingency retrain. Checkpoints land every 10k, so
# the 100k-vs-120k comparison is free. If the live curve still drops at 120k,
# extend by resuming — decide on the curve, not in advance.
STEPS="${STEPS:-120000}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -i $HOME/.ssh/id_ed25519"

stage() { echo; echo "=== $1 ==="; }

stage "0/7 preflight"
[ -f "$T/rp_key" ] || { echo "ABORT: no $T/rp_key"; exit 1; }
HF_TOKEN=$("$VENV/bin/python" -c "from huggingface_hub import get_token; print(get_token() or '')")
[ -n "$HF_TOKEN" ] || { echo "ABORT: no local HF token (hf auth login)"; exit 1; }
echo "rp_key ok, HF token ok"
echo "REMINDER: dataset pushed (push_dataset.py) and smoke_test.sh green? Ctrl-C within 5s if not."
sleep 5

stage "1/7 deploy (secure; fallback list — a failed try creates nothing and costs nothing)"
# Cheapest-adequate first, high-stock last. 16GB-class boxes are excluded: too
# slow for 150k x 2-cam inside the cost cap. Availability measured 07-21 23:20.
GPUS="${GPUS:-NVIDIA RTX A5000|NVIDIA RTX A4500|NVIDIA RTX 4000 Ada Generation|NVIDIA A40}"
POD=""
IFS='|' read -ra GPU_LIST <<< "$GPUS"
for GPU in "${GPU_LIST[@]}"; do
  echo "trying: $GPU"
  DEP=$("$VENV/bin/python" "$T/rp.py" deploy SECURE "$GPU" || true)
  POD=$(echo "$DEP" | "$VENV/bin/python" -c "
import sys, json
try: print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])
except Exception: print('')
")
  if [ -n "$POD" ]; then echo "deployed on: $GPU"; break; fi
  echo "  unavailable ($(echo "$DEP" | grep -oE 'SUPPLY_CONSTRAINT|[A-Z_]+_ERROR' | head -1 || echo no-capacity))"
done
[ -n "$POD" ] || { echo "ABORT: no GPU in the fallback list is available — try again later or edit GPUS"; exit 1; }
echo "$POD" > "$D/pod_id.txt"
echo "pod: $POD   (on ANY later failure: $VENV/bin/python $T/rp.py terminate $POD)"

stage "2/7 wait for SSH"
OUT=$("$VENV/bin/python" "$T/rp.py" waitssh "$POD")
echo "$OUT"
IP=$(echo "$OUT" | grep -oP 'ip=\K[0-9.]+'); PORT=$(echo "$OUT" | grep -oP 'port=\K[0-9]+')
echo "$IP $PORT" > "$D/pod_ssh.txt"

stage "3/7 baseline balance (real-spend accounting)"
"$VENV/bin/python" "$T/rp.py" list \
  | "$VENV/bin/python" -c "import sys,json;print(json.load(sys.stdin)['data']['myself']['clientBalance'])" \
  > "$D/baseline_balance.txt"
echo "baseline: \$$(cat "$D/baseline_balance.txt")"

stage "4/7 copy scripts to pod"
scp $SSH_OPTS -P "$PORT" "$T/cloud_setup.sh" "$T/train_act.sh" "$D/remote_run.sh" "root@$IP:/root/"

stage "5/7 cloud_setup.sh on pod (CUDA pre-gate + pinned install + hf auth)"
ssh $SSH_OPTS -p "$PORT" "root@$IP" "cd /root && HF_TOKEN=$HF_TOKEN bash cloud_setup.sh"

stage "6/7 start training (remote_run.sh, STEPS=$STEPS)"
ssh $SSH_OPTS -p "$PORT" "root@$IP" "STEPS=$STEPS bash /root/remote_run.sh"
echo "(verify from the log itself: the dashboard's step total must read $STEPS)"

stage "7/7 start watchdog + dashboard on the workcell"
if pgrep -f "run_v2/watchdog.sh" >/dev/null; then echo "watchdog already running"; else
  nohup bash "$D/watchdog.sh" >> "$D/watchdog_nohup.log" 2>&1 & echo "watchdog pid $!"
fi
if pgrep -f "run_v2/dashboard.py" >/dev/null; then echo "dashboard already running"; else
  nohup "$VENV/bin/python" "$D/dashboard.py" >> "$D/dashboard_nohup.log" 2>&1 & echo "dashboard pid $!"
fi

echo
echo "LAUNCHED. dashboard: http://localhost:8095  ·  status: cat $D/status.txt"
echo "watchdog auto-uploads ckpt-v2-last + terminates the pod on completion."
