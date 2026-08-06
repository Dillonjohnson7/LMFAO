#!/usr/bin/env python
"""Live browser view of the SO101 cameras — for aiming and FOCUSING them.

    python eval/live_view.py [--front DEV] [--wrist DEV] [--port N]

Serves http://<host>:8096/ with both cameras side by side (~15 fps MJPEG).
Each pane overlays a live SHARPNESS score (Laplacian variance of the center
region, box shown) — turn the focus ring to MAXIMIZE the number.

Defaults: wrist from wrist_cam.path, front from front_cam.path; --front/--wrist
override (use while a camera isn't pinned yet). A camera that drops shows
NO SIGNAL and auto-reopens — it doesn't kill the page.

⚠ UVC cams are single-reader: this HOLDS both cameras while it runs. Stop it
before recording or rollouts.
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import ROOT as _ROOT  # noqa: E402
BASE = str(_ROOT)
W, H, FPS = 1280, 720, 30
JPEG_Q = 80


def pinned(role):
    p = f"{BASE}/{role}_cam.path"
    if os.path.exists(p):
        return open(p).read().strip()
    return None


class Cam(threading.Thread):
    """Own one camera: capture, score sharpness, keep the latest JPEG."""

    def __init__(self, role, dev):
        super().__init__(daemon=True)
        self.role, self.dev = role, dev
        self.jpeg = None
        self.status = "starting"
        self._cap = None

    def _open(self):
        cap = cv2.VideoCapture(os.path.realpath(self.dev), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        return cap if cap.isOpened() else None

    def run(self):
        t_last, n, fps = time.perf_counter(), 0, 0.0
        while True:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self.status = "cannot open (busy/unplugged?) — retrying"
                    self._blank()
                    time.sleep(2)
                    continue
                self.status = "live"
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.status = "NO SIGNAL — reopening"
                self._blank()
                self._cap.release()
                self._cap = None
                time.sleep(1)
                continue
            n += 1
            now = time.perf_counter()
            if now - t_last >= 1.0:
                fps, n, t_last = n / (now - t_last), 0, now
            h, w = frame.shape[:2]
            # sharpness = Laplacian variance of the CENTER region (what you focus on)
            y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)
            for txt, org, scale in ((f"{self.role.upper()}  {self.dev}", (12, 34), 0.8),
                                    (f"sharpness {sharp:6.0f}   {fps:4.1f} fps", (12, h - 18), 1.1)):
                cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
                cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (60, 255, 60), 2)
            okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if okj:
                self.jpeg = buf.tobytes()

    def _blank(self):
        import numpy as np
        img = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(img, f"{self.role.upper()}: {self.status}", (40, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        okj, buf = cv2.imencode(".jpg", img)
        if okj:
            self.jpeg = buf.tobytes()


PAGE = """<!doctype html><html><head><title>SO101 live cams</title><style>
body{background:#111;color:#ddd;font-family:sans-serif;margin:12px}
div.row{display:flex;gap:12px;flex-wrap:wrap}
figure{margin:0}figcaption{font-size:18px;padding:4px 0}
img{max-width:48vw;min-width:400px;border:1px solid #333}
</style></head><body>
<h2>SO101 live cameras — turn the focus ring to MAXIMIZE the sharpness number</h2>
<div class=row>
<figure><figcaption>WRIST</figcaption><img src="/wrist.mjpg"></figure>
<figure><figcaption>FRONT (scene)</figcaption><img src="/front.mjpg"></figure>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default=pinned("front"))
    ap.add_argument("--wrist", default=pinned("wrist"))
    ap.add_argument("--port", type=int, default=8096)
    args = ap.parse_args()

    cams = {}
    for role, dev in (("wrist", args.wrist), ("front", args.front)):
        if dev is None:
            print(f"({role}: not pinned and not given — pane will read NO SIGNAL)")
            dev = f"/dev/nonexistent_{role}"
        cams[role] = Cam(role, dev)
        cams[role].start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            role = self.path.strip("/").removesuffix(".mjpg")
            cam = cams.get(role)
            if cam is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    buf = cam.jpeg
                    if buf is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(buf)}\r\n\r\n".encode())
                        self.wfile.write(buf)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on http://localhost:{args.port}/  (Ctrl-C to stop; cameras are HELD while this runs)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
