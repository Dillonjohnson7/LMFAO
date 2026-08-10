# NVIDIA Inception compute (Brev) — LMFAO training runbook

How to run the LMFAO / ACT training pipeline on our NVIDIA Inception Program
compute instead of RunPod. Written 2026-08-09 for the v2 training run
(`docs/RECIPE_V2.md`), valid for any future run.

## 1. What the Inception compute actually is

NVIDIA Inception issues **no compute directly** — it is a benefits portal that
routes member startups to partner compute programs. The main paths (2026):

| path | what you get | paradigm |
| --- | --- | --- |
| **NVIDIA Brev credits** ← *ours* | prepaid credit balance on brev.nvidia.com (NVIDIA's own GPU dev-cloud, aggregating 20+ providers) | self-serve GPU VMs, SSH, Docker preinstalled — the RunPod-shaped option |
| AWS Activate via Inception | up to $100K EC2 credits (tiers $10K/$25K/$100K) | raw EC2 GPU instances, you manage AMI/quotas |
| DGX Cloud Innovation Lab | 60-day hands-on DGX Cloud access (application-based) | DGX Cloud Lepton: batch jobs + dev pods, NGC containers |
| Nebius AI Lift | up to $150K credits | neocloud VMs |

We believe our grant is **Brev credits** (the Inception tile on
brev.nvidia.com). Confirm before doing anything else:

1. Log into the Inception portal → benefits catalog, look for the Brev /
   NVIDIA cloud credits tile, or check the acceptance email for a credit code.
2. If it turns out to be AWS Activate / DGX Cloud / Nebius instead, see §10 —
   the rest of this doc won't apply.

## 2. Redeeming the credits

1. Create an account at **brev.nvidia.com** (use the startup's business email;
   this creates your *organization*).
2. **Billing → Credits → Redeem a credit code.** The balance is shared across
   the whole org, so redeem in the right org before inviting anyone.
3. Teammates: **Team** tab → *Generate Invite Link* (links expire after 7
   days). Each member then sees the shared balance.
4. The console shows live balance + burn rate whenever instances are running.

Support channel for redemption problems: brev-support@nvidia.com.

## 3. The paradigm: RunPod → Brev mapping

| RunPod | Brev | notes |
| --- | --- | --- |
| pod | instance | GPU VM on one of 20+ providers; pick provider+GPU at creation |
| `/workspace` | `/home/ubuntu/workspace` | landing dir for `brev shell` / SSH |
| stop pod | stop instance | **no compute charges**, data preserved — but pinned to that provider/region; if capacity vanishes you can't restart until it returns. **Push work to git/HF before stopping.** |
| terminate | delete | data gone, unrecoverable |
| hourly bill | per-hour burn against the credit balance | storage charges may still apply while stopped |
| SSH key plumbing | managed by CLI | `brev shell <name>` or plain `ssh <name>` after `brev refresh` |

Instances come with NVIDIA drivers, CUDA, Python 3.10+, Docker, JupyterLab
preinstalled.

## 4. One-time local setup (Mac)

```bash
brew install brevdev/homebrew-brev/brev
brev --version
brev login            # browser OAuth; creates ~/.brev/ with SSH keys
```

(`brev login --token` for headless use; `brev login --skip-browser` prints the
URL instead of opening it.)

## 5. Launching the training instance

Our workload is tiny: ACT ~52M params, 100k steps, batch 8 — about
**4.3 h/policy at the v1 throughput (~6.5 steps/s)**, so the full v2 run
(stock + v2_steady + v2_jitter_occlusion) is **~13–15 GPU-hours**. Any modern
datacenter GPU is overkill; pick on price/availability.

- **Recommended: L40S** (48 GB, Ada) — closest analog to the 4090-class pods
  v1 trained on, usually the cheapest per hour. A100/H100 work unchanged if
  L40S capacity is short.
- Console path: brev.nvidia.com → **GPUs → Create**, pick provider/GPU, and
  paste this into the **setup script** field so the environment builds itself:

```bash
curl -LsSf https://raw.githubusercontent.com/Dillonjohnson7/LMFAO/main/scripts/brev_setup.sh | bash
```

- CLI path (equivalent):

```bash
brev create lmfao-train --gpu "nebius.l40sx1.pcie"   # provider string per console list
brev shell lmfao-train
```

- If you skip the setup-script field, run the same curl|bash line manually
  after first shell. It is idempotent.

`scripts/brev_setup.sh` reproduces the known-working environment
(`policytraining_v3.md` §9.2) exactly: Python 3.12 venv at
`~/envs/lerobot06`, torch 2.11.0+cu128 / torchvision 0.26.0+cu128, LeRobot
0.6.2 pinned to `ef88d4e`, PyAV 15.1.0, torchcodec 0.15.0, accelerate 1.14.0,
datasets 5.0.1, plus ffmpeg/tmux/rsync and a clone of this repo at
`~/workspace/LMFAO`. It ends with `nvidia-smi` + import smoke checks.

## 6. Data in (NUC → instance)

Datasets live on the NUC. Two transfer options — prefer rsync for
resumability:

```bash
# A. rsync over the Brev-managed SSH config (run `brev refresh` first so the
#    instance name resolves):
rsync -aP --exclude '._*' \
  multiply@100.103.79.98:/home/multiply/LMFAO/so101/datasets/eval_buckets_v4/ \
  lmfao-train:/home/ubuntu/workspace/data/eval_buckets_v4/

# B. or push from the NUC to the HF hub and `hf download` on the instance
#    (see policytraining_v3.md §10 for the hub layout).
```

Then **re-encode low-GOP on the instance copies** — the v1 long-GOP bottleneck
(§9.3) applies to any MP4 the augmenter writes, and this is a 10×
dataloader-throughput issue:

```bash
cd ~/workspace/LMFAO
~/envs/lerobot06/bin/python scripts/reencode_low_gop.py \
  ~/workspace/data/eval_buckets_v4/*/
```

## 7. Train

```bash
cd ~/workspace/LMFAO
tmux new -s train        # survive SSH drops; detach with Ctrl-b d

OUT_ROOT=$HOME/workspace/data/eval_buckets_v4 \
RUNS_ROOT=$HOME/workspace/runs/eval_buckets_v4 \
BUCKETS="stock v2_steady v2_jitter_occlusion" \
LEROBOT_BIN=$HOME/envs/lerobot06/bin/lerobot-train \
./scripts/train_eval_buckets.sh
```

Smoke first if the instance is fresh: `BUCKETS=stock STEPS=10 BATCH_SIZE=2
SAVE_CHECKPOINT=false ...`. Expect ~6.5 steps/s on an L40S once the low-GOP
re-encode has run; watch the first few hundred steps for throughput before
walking away.

## 8. Checkpoints out, then shut down

Checkpoints (~200 MB/policy) land in `$RUNS_ROOT/{bucket}`. **Extract before
stopping the instance** — stop preserves data but pins it to a provider whose
capacity may not be there when you come back:

```bash
# A. rsync back to the NUC:
rsync -aP lmfao-train:/home/ubuntu/workspace/runs/eval_buckets_v4/ \
  multiply@100.103.79.98:/home/multiply/LMFAO/so101/runs/eval_buckets_v4/

# B. or upload per the retention checklist (policytraining_v3.md §10.2).
```

Then `brev stop lmfao-train` (keeps disk, no compute burn) or
`brev delete lmfao-train` once everything is verified off-box.

## 9. Credit hygiene

- The console shows balance + burn rate; check it when you launch and when
  you stop.
- **Stop the instance whenever you are not actively training** — the whole
  point of credits is not to leak them on idle GPUs.
- Leave auto-recharge off unless you've attached a card deliberately; running
  out of credits pauses (then deletes) instances after a grace period.
- Rough budget: ~15 GPU-hours for the v2 run — trivial against any credit
  tier, but only if idle time isn't burning alongside.

## 10. If the grant turns out not to be Brev

- **AWS Activate via Inception**: Inception portal → benefits → AWS Activate
  tile; the portal shows a per-tier password ($10K/$25K/$100K tiers have
  *different* passwords) for the AWS application form. After AWS approves,
  everything moves to the AWS side (EC2 g6/p5, Deep Learning AMI, EBS). The
  `brev_setup.sh` pins still apply to whatever AMI you pick.
- **DGX Cloud / Innovation Lab**: apply via the Innovation Lab tile in the
  portal (rolling review). Paradigm is DGX Cloud Lepton: a workspace with
  node groups; training runs as **batch jobs** using an NGC PyTorch image
  (`nvcr.io/nvidia/pytorch:25.08-py3`) with a run command that pip-installs
  the pinned LeRobot env and invokes `train_eval_buckets.sh`; interactive
  work via dev pods.
- **Nebius AI Lift**: separate application through the portal; plain GPU VMs
  — this doc's §6–§8 transfer verbatim once you have SSH.

Either way, the environment pins (§5) and the data/training mechanics
(§6–§8) are platform-independent — only instance provisioning changes.
