#!/usr/bin/env bash
# ============================================================================
# Bootstrap a FRESH rented GPU box to train SO101 policies with lerobot.
# Tested target: Ubuntu 22.04/24.04 image that already has the NVIDIA driver +
# CUDA (the default on RunPod / Lambda / Vast / Paperspace GPU images).
#
# Run once per freshly-rented box:   bash cloud_setup.sh
# ============================================================================
set -euo pipefail

# --- 0. CUDA health gate BEFORE installing anything (Mistake #4) ------------
# The dud community 4090 had a healthy nvidia-smi but broken CUDA compute
# (err 999). A 2-second matmul with the IMAGE'S OWN torch catches that before
# we spend minutes on apt/uv/lerobot. RunPod pytorch images ship torch.
if python3 -c "import torch" 2>/dev/null; then
  python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "image torch sees no CUDA — dud box, terminate it"
x = torch.rand(1024, 1024, device="cuda")
torch.cuda.synchronize()
print("CUDA pre-gate OK:", torch.cuda.get_device_name(0), "| matmul:", (x @ x).sum().item())
PY
else
  echo "WARN: image has no torch — skipping pre-gate (final gate in step 5 still runs)"
fi

# root-safe: RunPod containers run as root with no `sudo` binary
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

# --- 1. system deps: ffmpeg (AV1 video decode), git, rsync (the workcell
# watchdog syncs logs/checkpoints via rsync — the RunPod pytorch image does NOT
# ship it; its absence silently killed every watchdog cycle on 07-21) ---------
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git ffmpeg curl rsync

# --- 2. uv (fast python + venv manager) ------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 3. python 3.12 venv (matches the workcell: 3.12) ----------------------
uv venv --python 3.12 "$HOME/lerobot-venv"
# shellcheck disable=SC1091
source "$HOME/lerobot-venv/bin/activate"

# --- 4. install the EXACT lerobot that recorded the data -------------------
# The dataset is codebase_version v3.0. The workcell runs lerobot 0.6.1, which
# is NOT on PyPI (PyPI stops at 0.6.0) -> pin the exact git commit so the cloud
# reads v3.0 identically. On a GPU box this pulls the CUDA torch build.
# NOTE the [training] extra: lerobot-train needs `accelerate` + `wandb`, which
# base lerobot does NOT install (verified — the run aborts without it). ACT is
# built into core; only pi0 later also needs the [pi] extra.
uv pip install \
  "lerobot[training] @ git+https://github.com/huggingface/lerobot.git@3f2179f3b69708b6ad009b2e7685dd9d05269ee1"

# --- 5. sanity: is the GPU visible to torch? -------------------------------
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__, "| cuda available:", ok,
      "|", (torch.cuda.get_device_name(0) if ok else "NO GPU — check the box/driver"))
assert ok, "CUDA not available — do not train here, it will fall back to CPU."
PY

# --- 6. authenticate to HF to pull the PRIVATE dataset ---------------------
# The dataset is private, so the box must authenticate.
#   - Non-interactive (driven over SSH): export HF_TOKEN=<token> before running.
#   - Interactive: leave HF_TOKEN unset and you'll be prompted.
# A READ token is enough to pull the dataset: https://huggingface.co/settings/tokens
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN"
else
  hf auth login
fi

echo
echo "Bootstrap complete."
echo "Next:  source ~/lerobot-venv/bin/activate  &&  bash train_act.sh"
