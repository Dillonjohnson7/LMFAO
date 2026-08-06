#!/usr/bin/env bash
# Host-side setup for SO101 on a Linux workcell: udev names + calibration files.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== SO101 host setup (LMFAO/so101) ==="

echo "Installing udev rule (needs sudo) ..."
sudo cp "$HERE/99-so101.rules" /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  installed /etc/udev/rules.d/99-so101.rules"

echo "Installing calibration files ..."
FOLL="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower"
LEAD="$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"
mkdir -p "$FOLL" "$LEAD"
cp "$HERE/follower.json" "$FOLL/follower.json"
cp "$HERE/leader.json" "$LEAD/leader.json"
echo "  installed follower.json + leader.json"

echo ""
echo "=== host setup done ==="
echo "Next:"
echo "  * Activate a LeRobot 0.6.x env with feetech + opencv extras"
echo "    (on the NUC: source /home/multiply/envs/lerobot06/bin/activate)"
echo "    or set LEROBOT_VENV=/path/to/venv"
echo "  * Pin cameras:  python $HERE/eval/cam_check.py --find-front && python $HERE/eval/cam_check.py --find"
echo "  * Record:       $HERE/rec 40"
echo "  * Verify:       ls -l /dev/so101_follower /dev/so101_leader"
