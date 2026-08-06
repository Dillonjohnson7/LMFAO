#!/bin/bash
# TRACKED BACKUP of ~/creep (the live copy) — sync after editing either. Added 07-22
# after the wrapper gained go2-10/30/100 modes that existed nowhere in git.
# ============================================================================
# SO101 policy rollout helper.  Run:  ~/creep [mode] [secs]
#
#   ~/creep            POSE mode — limp arm + live joint readout vs the demo
#                      start; ENTER locks the pose, Ctrl-C leaves it limp.
#   ~/creep dry        One-shot check, no motion: grounding + one inference.
#   ~/creep go 135     Baseline rollout, NSTEP=100 (the runs #13-15 config).
#   ~/creep go50 135   Rollout at NSTEP=50 — one mid-descent correction.
#   ~/creep go30 135   Rollout at NSTEP=30 — fallback if 50 still pecks blind.
#   ~/creep retry 135  NSTEP=50 + close-on-air AUTO-RETRY (re-home to demo
#                      start with alternating pan offsets, max 3).
#   ~/creep retry30    Same auto-retry at NSTEP=30.
#   ~/creep check      SELFTEST — no robot, no motion, safe any time.
#
#   v2 (wrist-cam policy, after the retrain — see V2_PIPELINE.md):
#   ~/creep dry2 / go2 135   same as dry/go but front+wrist cams, v2 model+stats.
#
# TONIGHT'S SERIES (NEXT_STEPS.md item 1):
#   1) 3x  ~/creep go50 135    puck at the demo-mean spot each run
#   2) if grasps >= the 1/3 baseline: 3x  ~/creep retry 135
#   Score from the log verdicts (HOLD / close-on-air) + frames in
#   eval/run_recordings/; grip <20 = air, 26-36 = held. Update CHECKPOINT.md.
#
# In go/retry modes Ctrl-C stops INSTANTLY — the arm holds pose (never drops).
# Guardrails: per-motor deg/tick clamp, demo min-max envelope, start-pose gate
# (>25 deg off => refuses to move), bus watchdog, hard time cap.
# Config is BAKED here (Mistake #12): never rely on env prefixes typed at the
# terminal — this machine's paste bug silently drops them.
# ============================================================================
cd /home/anvil/SO101_policy/eval
PY=/home/anvil/SO101_policy/.venv/bin/python
LOG=/home/anvil/SO101_policy/eval/creep_runs.log
# Policy under evaluation: the 150k checkpoint (override with MODEL=... ~/creep ...)
export MODEL="${MODEL:-/home/anvil/SO101_policy/training/run150k/checkpoints/last/pretrained_model}"
# v2 defaults (exist only after the V2_PIPELINE.md retrain)
V2_MODEL="${MODEL2:-/home/anvil/SO101_policy/training/run_v2/checkpoints/last/pretrained_model}"
V2_ROOT=/home/anvil/SO101_policy/datasets/pick_place_v2

run() { # run <nstep> <retry> <secs>
  FAST=1 NSTEP="$1" RETRY="$2" DURATION="$3" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG"
}
# Pre-flight: the ALAN sentinel containers own the arm + front cam while they
# run — every hardware mode fails confusingly until they are stopped.
if [ "$1" != "check" ] && docker ps --format '{{.Names}}' 2>/dev/null | grep -qE '^so101-(camera|sentinel-web)$'; then
  echo "⚠  ALAN sentinel containers are RUNNING — they hold the arm/front cam."
  echo "   Stop them first (your call):  docker stop so101-camera so101-sentinel-web"
  echo ""
fi
case "$1" in
  go)      run 100 0 "${2:-90}"  ;;
  go50)    run 50  0 "${2:-135}" ;;
  go30)    run 30  0 "${2:-135}" ;;
  retry)   run 50  1 "${2:-135}" ;;
  retry30) run 30  1 "${2:-135}" ;;
  slow)    FAST=0 NSTEP=10 DURATION="${2:-45}" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG" ;;
  dry)     exec "$PY" -u creep_test.py ;;
  dry2)    CAMS=front,wrist MODEL="$V2_MODEL" DATASET_ROOT="$V2_ROOT" DATASET_REPO=local/pick_place_v2 \
             exec "$PY" -u creep_test.py ;;
  # go2-30 = go2 with NSTEP=30: one extra mid-descent re-plan — lets the policy
  # use the wrist view for a late close correction (the 07-22 bite-scatter test).
  go2-30)  CAMS=front,wrist MODEL="$V2_MODEL" DATASET_ROOT="$V2_ROOT" DATASET_REPO=local/pick_place_v2 \
             FAST=1 NSTEP=30 RETRY=0 DURATION="${2:-135}" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG" ;;
  go2-100) CAMS=front,wrist MODEL="$V2_MODEL" DATASET_ROOT="$V2_ROOT" DATASET_REPO=local/pick_place_v2 \
             FAST=1 NSTEP=100 RETRY=0 DURATION="${2:-135}" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG" ;;
  go2-10)  CAMS=front,wrist MODEL="$V2_MODEL" DATASET_ROOT="$V2_ROOT" DATASET_REPO=local/pick_place_v2 \
             FAST=1 NSTEP=10 RETRY=0 DURATION="${2:-135}" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG" ;;
  # go2 = a CLEAN v2 rollout: NSTEP=50, front+wrist, basic guardrails only.
  # NO auto-retry / jam detection (user call 07-21: that apparatus was an item-1
  # band-aid for v1's blindness — v2's wrist cam is the real fix; evaluate it
  # unconfounded). The grip verdict lines still print (passive scoring telemetry).
  go2)     CAMS=front,wrist MODEL="$V2_MODEL" DATASET_ROOT="$V2_ROOT" DATASET_REPO=local/pick_place_v2 \
             FAST=1 NSTEP=50 RETRY=0 DURATION="${2:-135}" "$PY" -u creep_test.py --go 2>&1 | tee -a "$LOG" ;;
  check)   NSTEP=50 exec "$PY" -u creep_test.py --selftest ;;
  *)       exec "$PY" -u creep_test.py --pose ;;
esac
