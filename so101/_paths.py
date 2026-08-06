"""Repo-relative paths for the SO101 workcell tools under LMFAO/so101.

Override with SO101_ROOT / DATASET_ROOT / FRONT_CAM_FILE / WRIST_CAM_FILE.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SO101_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
DATASETS = ROOT / "datasets"
DEFAULT_DATASET = Path(
    os.environ.get("DATASET_ROOT", str(DATASETS / "pick_place_v3"))
).expanduser()
FRONT_CAM_FILE = Path(os.environ.get("FRONT_CAM_FILE", str(ROOT / "front_cam.path"))).expanduser()
WRIST_CAM_FILE = Path(os.environ.get("WRIST_CAM_FILE", str(ROOT / "wrist_cam.path"))).expanduser()
CALIB_DIR = ROOT / "calib"
