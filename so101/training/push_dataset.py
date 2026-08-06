#!/usr/bin/env python
"""Push a local LeRobot dataset to the HF Hub (canonical meta/data/videos only,
never the images/ recorder scratch), then (re)point the revision tag at it.

  python push_dataset.py                       # datasets/pick_place_v2 -> Dillonjohnson/pick_place_v2
  python push_dataset.py --root ... --repo ... --tag v3.0

The tag matters: train_act.sh / the cloud box pull by repo id; the tag records
which snapshot trained which model (v1 used tag v3.0 = the lerobot codebase
version of the on-disk format).
Requires Hub auth (a WRITE token):  hf auth login
"""
import argparse
import os
import sys

os.environ.pop("HF_HUB_OFFLINE", None)          # this script exists to go online
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/SO101_policy/datasets/pick_place_v2"))
ap.add_argument("--repo", default="Dillonjohnson/pick_place_v2")
ap.add_argument("--tag", default="v3.0")
a = ap.parse_args()

if not os.path.exists(os.path.join(a.root, "meta", "tasks.parquet")):
    sys.exit(f"no finalized dataset at {a.root} — record demos first (~/rec)")

api = HfApi()
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
print(f"uploading {a.root} -> {a.repo} (meta/ data/ videos/ only) ...")
api.upload_folder(
    folder_path=a.root, repo_id=a.repo, repo_type="dataset",
    allow_patterns=["meta/*", "data/*", "videos/*"],   # hub glob: * crosses '/'
    # MIRROR the local tree: the 07-21 episode-delete rewrote/renumbered files,
    # and a stale Hub file the local meta no longer references would poison the
    # cloud pull-train. delete_patterns removes anything not re-uploaded.
    delete_patterns=["meta/*", "data/*", "videos/*"],
)
try:
    api.delete_tag(a.repo, tag=a.tag, repo_type="dataset")
except Exception:
    pass                                         # tag didn't exist yet — fine
api.create_tag(a.repo, tag=a.tag, repo_type="dataset")
print(f"done: https://huggingface.co/datasets/{a.repo} (tag {a.tag})")
