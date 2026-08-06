#!/usr/bin/env python
"""Puck-placement aimer (v2). (Filename is historical — the object was once
mis-called a "ring"; it is a ~7 cm BLUE PUCK, user-settled 07-22.) Run between rollouts (front camera must be free):

    python eval/ring_spot.py

Grabs a live front-cam frame, draws the v2 demo-mean target circle, saves
eval/cam_probe/ring_spot.jpg. Nudge the puck, re-run, until the puck fills the
circle. Also prints the measured offset if it can find the blue puck.

v2 numbers (derived 2026-07-22 from frame 0 of all 36 clean-pick demos,
overlay: eval/cam_probe/v2_ring_targets.jpg): mean center (756,584),
placement spread sigma (31,26) px, puck radius 21.8 px -> ~6.2 px/cm.
The front cam is the repurposed Anvil 4K (pinned in front_cam.path) — it
renders the puck far less saturated than the v1 Sonix did, hence the low
S floor; detection is confined to the table ROI to reject the tissue box
and anything left in the bin."""
import os

import cv2
import numpy as np

TARGET = (756, 584)          # mean puck start across the 36 clean-pick v2 demos
RADIUS = 22                  # ~puck outer radius in px at that depth
ROI = (650, 950, 480, 700)   # x0,x1,y0,y1 — the demos' puck start zone
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import FRONT_CAM_FILE as _FCF, ROOT as _ROOT  # noqa: E402
PIN = str(_FCF)
OUT = str(_ROOT / "eval" / "cam_probe" / "ring_spot.jpg")

if not os.path.exists(PIN):
    raise SystemExit("front camera not pinned — run: python eval/cam_check.py --find-front")
CAM = os.path.realpath(open(PIN).read().strip())

cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok = False
for _ in range(10):                      # warm up exposure
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("camera busy or unreadable — is a rollout (or the sentinel) running?")

# find the blue puck — v2 thresholds (Anvil 4K desaturates the puck: S median
# ~53 where the Sonix gave >120), searched only inside the table ROI
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (95, 25, 80), (135, 255, 255))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
roi = np.zeros_like(mask)
roi[ROI[2]:ROI[3], ROI[0]:ROI[1]] = 255
mask &= roi
found = None
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    # size band, both ends: daylight gives the whole scene a blue cast that
    # forms huge low-S blobs (seen 07-22: area 59k, r 184 — the table shadow);
    # a real puck is area ~600-1500, enclosing r ~19-25 at this depth
    area = cv2.contourArea(c)
    (_, _), r = cv2.minEnclosingCircle(c)
    # circularity: the puck reads as a filled disc (demo sweep: ~0.67); the
    # daylight-cast blobs are sparse/irregular (~0.2)
    circ = area / (np.pi * r * r) if r > 0 else 0
    if 300 < area < 3000 and 14 < r < 32 and circ > 0.45:
        m = cv2.moments(c)
        found = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        break

cv2.rectangle(frame, (ROI[0], ROI[2]), (ROI[1], ROI[3]), (255, 255, 0), 1)
cv2.circle(frame, TARGET, RADIUS, (0, 255, 0), 3)
cv2.circle(frame, TARGET, 4, (0, 255, 0), -1)
cv2.putText(frame, "put the puck IN the green circle", (TARGET[0] - 180, TARGET[1] - RADIUS - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
msg = "puck not detected (in the table ROI)"
if found:
    dx, dy = found[0] - TARGET[0], found[1] - TARGET[1]
    cv2.circle(frame, found, 6, (0, 0, 255), -1)
    cv2.arrowedLine(frame, found, TARGET, (0, 0, 255), 2)
    cm = 7.0 / (2 * RADIUS)              # ~cm per px at puck depth (~0.16)
    # ON-TARGET tolerance ~1.6 cm, same physical slack the v1 aimer used
    tol_px = 10
    msg = (f"puck is {abs(dx)*cm:.1f}cm {'LEFT' if dx < 0 else 'RIGHT'} "
           f"+ {abs(dy)*cm:.1f}cm {'FAR' if dy < 0 else 'NEAR'} of target"
           if (abs(dx) > tol_px or abs(dy) > tol_px) else "ON TARGET - go")
cv2.putText(frame, msg, (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, frame)
print(msg)
print(f"open {OUT}")
