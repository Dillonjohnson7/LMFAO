#!/usr/bin/env bash
# brev_setup.sh — one-shot environment setup for LMFAO ACT training on an
# NVIDIA Brev GPU instance. Reproduces the known-working training environment
# (docs/policytraining_v3.md §9.2) on a fresh Brev VM.
#
# Run it on the instance:
#   curl -LsSf https://raw.githubusercontent.com/Dillonjohnson7/LMFAO/main/scripts/brev_setup.sh | bash
# or paste that same URL into the instance's setup-script field at creation
# (Brev runs it automatically), or run ./scripts/brev_setup.sh from a clone.
#
# Idempotent: safe to re-run; existing venv/repo are reused.
set -euo pipefail

LEROBOT_COMMIT="ef88d4e52b9f3a16638e4b73202d619b7606fd41"
VENV="${LEROBOT_VENV:-$HOME/envs/lerobot06}"
WORKSPACE="${WORKSPACE:-$HOME/workspace}"

echo "=== system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg git curl tmux rsync

echo "=== uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== python 3.12 venv at ${VENV} ==="
uv python install 3.12
if [[ ! -d "$VENV" ]]; then
  uv venv --python 3.12 "$VENV"
fi

# Torch first so lerobot's resolver keeps the cu128 build (PEP 440 local
# segment satisfies lerobot's torch requirement).
echo "=== torch 2.11.0+cu128 / torchvision 0.26.0+cu128 ==="
uv pip install --python "$VENV/bin/python" \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== lerobot 0.6.2 @ ${LEROBOT_COMMIT:0:7} (pinned) ==="
uv pip install --python "$VENV/bin/python" \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

echo "=== pinned runtime deps (policytraining_v3.md §9.2) ==="
uv pip install --python "$VENV/bin/python" \
  "av==15.1.0" "torchcodec==0.15.0" "accelerate==1.14.0" "datasets==5.0.1"

echo "=== LMFAO repo (training scripts) ==="
mkdir -p "$WORKSPACE"
if [[ ! -d "$WORKSPACE/LMFAO/.git" ]]; then
  git clone https://github.com/Dillonjohnson7/LMFAO.git "$WORKSPACE/LMFAO"
else
  git -C "$WORKSPACE/LMFAO" pull --ff-only
fi

echo "=== smoke checks ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$VENV/bin/python" - <<'PY'
import torch, av, lerobot
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("av", av.__version__)
print("lerobot", lerobot.__version__)
PY
"$VENV/bin/lerobot-train" --help >/dev/null && echo "lerobot-train OK"

echo
echo "Done. Activate with:  source ${VENV}/bin/activate"
echo "Next: docs/INCEPTION_BREV.md §6 (data transfer)"
