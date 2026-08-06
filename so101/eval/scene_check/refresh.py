#!/usr/bin/env python
"""Regenerate the scene-check page for the v2 (2-cam) policy.

    python eval/scene_check/refresh.py            # grab live front+wrist frames only
    python eval/scene_check/refresh.py --thumbs   # also re-sample training thumbnails

Open eval/scene_check/index.html afterwards. Run before every rollout session:
the pre-go2 gate is that BOTH live frames look like their training rows —
same framing, same lighting temperature (daylight gives a strong blue cast:
B/R was 1.27 at 10:40 vs 0.94 in the evening demos — measured 07-22).
"""
import os, random, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2

D = os.path.dirname(os.path.abspath(__file__))
PINS = {"front": "/home/anvil/SO101_policy/front_cam.path",
        "wrist": "/home/anvil/SO101_policy/wrist_cam.path"}
CAMS = ["front", "wrist"]
N_THUMBS = 3

def grab_live():
    for name in CAMS:
        p = os.path.realpath(open(PINS[name]).read().strip())
        cap = cv2.VideoCapture(p, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok = False
        for _ in range(10):
            ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"{name} camera busy or unreadable")
        cv2.imwrite(f"{D}/{name}_live.jpg", frame)
        print(f"{name}_live.jpg written")

def make_thumbs():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/pick_place_v2",
                        root="/home/anvil/SO101_policy/datasets/pick_place_v2",
                        video_backend="pyav")
    eps = ds.meta.episodes
    picks = []
    for k in range(N_THUMBS):
        ep = random.randrange(36)              # clean-pick episodes only
        i0, i1 = int(eps["dataset_from_index"][ep]), int(eps["dataset_to_index"][ep])
        i = random.randrange(i0, i1)
        item = ds[i]
        for name in CAMS:
            img = item[f"observation.images.{name}"]
            bgr = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype("uint8"),
                               cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{D}/{name}_t{k}.jpg", bgr)
        picks.append((ep, i - i0))
        print(f"thumb {k}: episode {ep}, frame {i - i0}")
    rows = []
    for name in CAMS:
        figs = [f'<figure><img src="{name}_live.jpg"><figcaption>▶ LIVE — {name} cam right now</figcaption></figure>']
        figs += [f'<figure><img src="{name}_t{k}.jpg"><figcaption>training · ep {ep}, frame {f}</figcaption></figure>'
                 for k, (ep, f) in enumerate(picks)]
        rows.append(f'<h2>{name} cam</h2><div class="t">{"".join(figs)}</div>')
    html = ('<!doctype html><meta charset="utf-8"><title>v2 scene check</title>'
            '<style>body{background:#111;color:#eee;font-family:sans-serif;margin:16px}'
            'div.t{display:flex;gap:8px;flex-wrap:wrap}figure{margin:0}'
            'img{width:420px;display:block}figcaption{font-size:13px;padding:2px}</style>'
            '<h1>v2 scene check — live vs pick_place_v2 training frames</h1>'
            '<p>refresh live frames: <code>python eval/scene_check/refresh.py</code></p>'
            + "".join(rows))
    with open(f"{D}/index.html", "w") as f:
        f.write(html)
    print("index.html rewritten (2 rows: front, wrist)")

grab_live()
if "--thumbs" in sys.argv:
    make_thumbs()
