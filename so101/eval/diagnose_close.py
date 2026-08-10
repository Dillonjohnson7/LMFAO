#!/usr/bin/env python3
"""Diagnose the missed close: joint-pose analysis of demos vs rollout episodes.

For each episode, find the grasp-close event (first large gripper command
deviation from its initial value), then record the arm joint pose (state
dims 0..4) at that frame. Compare:
  - demo close-pose spread across the 40 training demos
  - rollout close poses per run, split by success/failure
  - whether failures miss in a consistent direction (systematic bias)
    or scatter randomly (variance)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BASE = Path(os.environ.get("SO101_ROOT", Path(__file__).resolve().parents[1]))
DEMO = BASE / "datasets" / "pick_place_v3"
RUNS = {
    "stock_scored": (BASE / "eval/run_recordings/2026-08-06_1833_stock", {0, 1, 3, 13}),
    "stock_unscored": (BASE / "eval/run_recordings/2026-08-06_2023_stock", {0, 1, 2}),
    "occlusion": (BASE / "eval/run_recordings/2026-08-06_1928_occlusion", {2, 10, 11, 12, 14}),
    "full": (BASE / "eval/run_recordings/2026-08-06_2034_full", {0, 5, 9, 10, 14}),
}

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "wrist_roll"]


def load_episodes(root):
    out = {}
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        return out
    for f in files:
        cols = ["episode_index", "frame_index", "observation.state", "action"]
        try:
            d = pq.read_table(f, columns=cols).to_pydict()
        except Exception as e:
            print(f"  !! {f.name}: {e}")
            return out
        eps = np.array(d["episode_index"])
        fidx = np.array(d["frame_index"])
        states = np.array(d["observation.state"], dtype=float)
        actions = np.array(d["action"], dtype=float)
        for ep in np.unique(eps):
            m = eps == ep
            o = np.argsort(fidx[m])
            out[int(ep)] = (states[m][o], actions[m][o])
    return out


def close_event(states, actions):
    """Index of the grasp close: first frame where the gripper command has
    moved >50% of its episode range away from its initial value."""
    g = actions[:, -1]
    rng = g.max() - g.min()
    if rng < 1e-6:
        return None
    thresh = g[0] + np.sign(g[-1] - g[0]) * 0  # unused
    dev = np.abs(g - g[:50].mean())
    idx = np.where(dev > 0.5 * rng)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def analyze(name, root, fail_eps=None):
    eps = load_episodes(root)
    if not eps:
        print(f"[{name}] no data at {root}")
        return None
    rows = []
    for ep, (states, actions) in sorted(eps.items()):
        ci = close_event(states, actions)
        if ci is None:
            print(f"[{name}] ep{ep}: no close detected")
            continue
        pose = states[ci, :5]
        rows.append({
            "ep": ep,
            "close_frame": ci,
            "close_time_s": round(ci / 30.0, 1),
            "pose": pose,
            "fail": (fail_eps is not None and ep in fail_eps),
        })
    print(f"\n=== {name} ({len(rows)} episodes) ===")
    if not rows:
        return None
    P = np.array([r["pose"] for r in rows])
    times = [r["close_time_s"] for r in rows]
    print(f"close time: mean {np.mean(times):.1f}s  range [{min(times)}, {max(times)}]")
    print("per-joint close-pose stats (units of state vector):")
    for j, jn in enumerate(JOINTS[: P.shape[1]]):
        print(f"  {jn:14s} mean {P[:, j].mean():8.3f}  std {P[:, j].std():6.3f}  "
              f"range [{P[:, j].min():8.3f}, {P[:, j].max():8.3f}]")
    if fail_eps is not None:
        F = np.array([r["pose"] for r in rows if r["fail"]])
        S = np.array([r["pose"] for r in rows if not r["fail"]])
        print(f"successes: {len(S)}  failures: {len(F)}")
        if len(F) and len(S):
            d = F.mean(0) - S.mean(0)
            pooled = np.sqrt((S.var(0) * len(S) + F.var(0) * len(F)) / (len(S) + len(F)))
            print("failure minus success mean pose (per joint):")
            for j, jn in enumerate(JOINTS[: P.shape[1]]):
                t = d[j] / (np.sqrt(S[:, j].var() / len(S) + F[:, j].var() / len(F)) + 1e-9)
                print(f"  {jn:14s} delta {d[j]:+8.3f}  (pooled std {np.sqrt(pooled[j]):6.3f}, t={t:+5.2f})")
            # per-failure direction vs success mean: systematic if all same sign
            print("per-failure delta vs success mean (sign pattern):")
            for r in rows:
                if r["fail"]:
                    dd = r["pose"] - S.mean(0)
                    signs = " ".join(f"{jn[:2]}={x:+.2f}" for jn, x in zip(JOINTS, dd))
                    print(f"  ep{r['ep']:02d} t={r['close_time_s']:5.1f}s  {signs}")
    return rows


def main():
    demo_rows = analyze("demos pick_place_v3", DEMO)
    all_rows = {}
    for name, (root, fails) in RUNS.items():
        r = analyze(name, root, fails)
        if r:
            all_rows[name] = r
    if demo_rows:
        D = np.array([r["pose"] for r in demo_rows])
        print("\n=== rollout close pose vs DEMO mean (systematic offset check) ===")
        dmean = D.mean(0)
        dstd = D.std(0)
        for name, rows in all_rows.items():
            P = np.array([r["pose"] for r in rows])
            off = P.mean(0) - dmean
            z = off / (dstd + 1e-9)
            print(f"[{name}] mean offset vs demo mean, in demo-std units:")
            print("   " + "  ".join(f"{jn[:2]} {o:+.3f} ({zz:+.2f}σ)" for jn, o, zz in zip(JOINTS, off, z)))


if __name__ == "__main__":
    main()
