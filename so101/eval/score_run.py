#!/usr/bin/env python
"""Label the newest unscored rollout recording as OK (task success) or NG (fail).

    python eval/score_run.py OK|NG [run_number]

Renames eval/run_recordings/run_<ts>/ -> {OK|NG}_run<N>_<ts>/ and encodes
labeled videos next to it: {OK|NG}_run<N>_front.mp4 (+ _wrist.mp4 when wrist
frames exist — recorded from run #22 on). N comes from .run_counter (+1) unless
given explicitly; the counter is updated either way. Only dirs from 2026-07-22
onward are eligible — the older run_2026072* dirs are the v1 evidence base
that CHECKPOINT.md's footage index points into; never rename those."""
import glob
import os
import sys

D = "/home/anvil/SO101_policy/eval/run_recordings"
COUNTER = os.path.join(D, ".run_counter")
EARLIEST = "run_20260722"          # v1-era dirs are frozen evidence

if len(sys.argv) < 2 or sys.argv[1].upper() not in ("OK", "NG"):
    sys.exit("usage: score_run.py OK|NG [run_number]")
label = sys.argv[1].upper()

cands = sorted(d for d in glob.glob(os.path.join(D, "run_*"))
               if os.path.isdir(d) and os.path.basename(d) >= EARLIEST)
if not cands:
    sys.exit("no unscored recording found (already labeled? v1 dirs are excluded)")
src = cands[-1]
ts = os.path.basename(src)[len("run_"):]

if len(sys.argv) > 2:
    n = int(sys.argv[2])
else:
    n = (int(open(COUNTER).read().strip()) if os.path.exists(COUNTER) else 21) + 1
dst = os.path.join(D, f"{label}_run{n}_{ts}")
os.rename(src, dst)
with open(COUNTER, "w") as f:
    f.write(str(n))
print(f"labeled: {os.path.basename(dst)}")

import av
import cv2

for prefix, cam in (("f", "front"), ("w", "wrist")):
    frames = sorted(glob.glob(os.path.join(dst, f"{prefix}*.jpg")))
    if not frames:
        print(f"({cam}: no frames — recorded pre-07-22?)" if cam == "wrist" else f"({cam}: no frames)")
        continue
    out = os.path.join(D, f"{label}_run{n}_{cam}.mp4")
    c = av.open(out, "w")
    st = c.add_stream("h264", rate=6)
    h, w = cv2.imread(frames[0]).shape[:2]
    st.height, st.width = h, w
    st.pix_fmt = "yuv420p"
    st.options = {"crf": "23"}
    for fp in frames:
        vf = av.VideoFrame.from_ndarray(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), format="rgb24")
        for pkt in st.encode(vf):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    print(f"encoded: {os.path.basename(out)} ({len(frames)} frames)")
