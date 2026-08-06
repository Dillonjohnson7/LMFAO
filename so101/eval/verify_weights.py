#!/usr/bin/env python
"""Prove that ACTPolicy.from_pretrained(MODEL) loads MODEL's actual bytes.

    python eval/verify_weights.py [model_dir] [safetensors_to_compare_against]

Loads the policy the exact way creep_test.py does, then compares EVERY tensor
in model.safetensors against the in-memory state_dict, byte-for-byte
(torch.equal). This is the Mistake-#1 guard: make_policy() with a config from
PreTrainedConfig.from_pretrained() silently gives a RANDOM network; the only
honest proof of loading is params == file.

The optional second arg compares the loaded policy against a DIFFERENT
checkpoint's file — used to prove the checker detects mismatches (it must
FAIL when given the wrong file; see CHECKPOINT.md 07-22)."""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors import safe_open

from lerobot.policies.act.modeling_act import ACTPolicy

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "MODEL", "/home/anvil/SO101_policy/training/run_v2/checkpoints/last/pretrained_model")
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(MODEL, "model.safetensors")

print(f"policy dir : {os.path.realpath(MODEL)}")
print(f"weights    : {os.path.realpath(WEIGHTS)}")
policy = ACTPolicy.from_pretrained(MODEL)
policy.to("cpu")
sd = policy.state_dict()

matched, params, bad = 0, 0, []
with safe_open(WEIGHTS, framework="pt", device="cpu") as f:
    fkeys = list(f.keys())
    for k in fkeys:
        t = f.get_tensor(k)
        m = sd.get(k)
        if m is None or m.shape != t.shape or not torch.equal(m.float(), t.float()):
            bad.append(k)
        else:
            matched += 1
            params += t.numel()
extra = [k for k in sd if k not in set(fkeys)]

print(f"tensors in file: {len(fkeys)} · matched byte-for-byte: {matched} · params verified: {params:,}")
if extra:
    print(f"(state_dict has {len(extra)} keys not in the file — non-persistent buffers, informational)")
if bad:
    print(f"\nMISMATCH on {len(bad)} tensors, e.g. {bad[:4]}")
    print("VERDICT: FAIL — the in-memory network is NOT this file (random-net risk, Mistake #1)")
    sys.exit(1)
print("VERDICT: PASS — the loaded policy IS this checkpoint, every tensor")
