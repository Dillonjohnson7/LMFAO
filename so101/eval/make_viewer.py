#!/usr/bin/env python
"""Split a recorded series into per-episode clips and build a page to watch them.

    python so101/eval/make_viewer.py so101/eval/run_recordings/<stamp>_<policy>

Writes, inside the run directory:

    web/index.html          all episodes, autoplaying and looping in sync
    web/clips/epNN.mp4      one clip per episode
    clips/trialNN_epNN_*    the same files, hardlinked under descriptive names

The shards are concatenated before cutting. LeRobot rolls the video over to a
new file mid-series, and an episode can straddle that boundary -- cutting from
whichever shard holds an episode's first frame yields a clip one frame long for
exactly that episode, which shows up as a black tile.

Verdicts come from review/summary.md when it is there, so the labels on the page
are the ones the scorer actually produced rather than a second opinion.
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1]).expanduser().resolve()
    cam = sys.argv[2] if len(sys.argv) > 2 else "front"
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))

    t = pd.concat([pq.read_table(f).to_pandas()
                   for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))],
                  ignore_index=True)
    ep = t["episode_index"].to_numpy()
    eps = sorted(set(ep.tolist()))

    shards = sorted(glob.glob(str(root / "videos" / f"observation.images.{cam}" / "**" / "*.mp4"),
                              recursive=True))
    if not shards:
        sys.exit(f"no {cam} video under {root}")

    web = root / "web"
    clips = web / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    joined = shards[0]
    tmp = None
    if len(shards) > 1:
        tmp = web / "_joined.mp4"
        lst = web / "_concat.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in shards))
        print(f"joining {len(shards)} shards…")
        run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(tmp)])
        joined = str(tmp)
        lst.unlink()

    print(f"cutting {len(eps)} episodes…")
    for e in eps:
        s = np.where(ep == e)[0]
        lo, hi = int(s[0]), int(s[-1])
        run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-ss", f"{lo/fps:.3f}", "-t", f"{(hi-lo+1)/fps:.3f}",
             "-i", joined, "-c", "copy", str(clips / f"ep{e:02d}.mp4")])

    # A clip that decodes to nothing is the shard-boundary bug returning; catch
    # it here rather than letting it show up as a black tile in the browser.
    bad = []
    for e in eps:
        p = clips / f"ep{e:02d}.mp4"
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
        try:
            dur = float(out)
        except ValueError:
            dur = 0.0
        if dur < 5.0:
            bad.append((e, dur))
    if bad:
        sys.exit(f"clips too short, cut is wrong: {bad}")
    print(f"  all {len(eps)} clips >= 5 s")

    if tmp and tmp.exists():
        tmp.unlink()

    # Verdicts as the scorer recorded them.
    failed: set[int] = set()
    summary = root / "review" / "summary.md"
    if summary.exists():
        for line in summary.read_text().splitlines():
            m = re.match(r"\|\s*(\d+)\s*\|", line)
            if m and "FAIL" in line:
                failed.add(int(m.group(1)))
        print(f"  verdicts from review/summary.md — failures: {sorted(failed)}")
    else:
        print("  no review/summary.md; every tile will read 'unscored'")

    # Descriptive hardlinks, so the clips are usable outside the page without
    # a second copy of a few hundred MB.
    named = root / "clips"
    named.mkdir(exist_ok=True)
    for e in eps:
        v = "failed" if e in failed else ("placed" if summary.exists() else "unscored")
        dst = named / f"trial{e+1:02d}_ep{e:02d}_{v}.mp4"
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(clips / f"ep{e:02d}.mp4")
        except Exception:
            shutil.copy2(clips / f"ep{e:02d}.mp4", dst)

    n_pass = len(eps) - len(failed)
    title = root.name
    (web / "index.html").write_text(PAGE
        .replace("__TITLE__", title)
        .replace("__CAM__", cam)
        .replace("__N__", str(len(eps)))
        .replace("__SCORE__", f"{n_pass}/{len(eps)} placed the puck" if summary.exists() else "unscored")
        .replace("__FAILED__", json.dumps(sorted(failed))))
    print(f"\n{web / 'index.html'}")


PAGE = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { margin:0; padding:12px; background:#111; color:#ddd;
         font:13px/1.4 ui-monospace, Menlo, Consolas, monospace; }
  h1 { font-size:15px; font-weight:600; margin:0 0 4px; }
  .sub { color:#888; margin-bottom:10px; }
  .bar { position:sticky; top:0; background:#111; padding:8px 0 10px;
         border-bottom:1px solid #333; margin-bottom:12px; z-index:2; }
  button { font:inherit; background:#262626; color:#ddd; border:1px solid #444;
           padding:5px 12px; margin-right:6px; cursor:pointer; border-radius:3px; }
  button:hover { background:#333; }
  #seek { width:320px; vertical-align:middle; }
  #t { color:#888; margin-left:8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:10px; }
  figure { margin:0; }
  video { width:100%; display:block; background:#000; border:2px solid #333; }
  figcaption { padding:3px 1px; display:flex; justify-content:space-between; }
  .pass { color:#4ade80; } .fail { color:#f87171; }
  .pass-b { border-color:#185b32; } .fail-b { border-color:#7f1d1d; }
</style>

<h1>__TITLE__ — __CAM__ camera</h1>
<div class="sub">__SCORE__ · looping in sync</div>

<div class="bar">
  <button onclick="play()">play</button>
  <button onclick="pauseAll()">pause</button>
  <button onclick="restart()">restart</button>
  <input id="seek" type="range" min="0" max="56.7" step="0.1" value="0" oninput="seekAll(+this.value)">
  <span id="t">0.0s</span>
</div>

<div class="grid" id="grid"></div>

<script>
  const FAILED = new Set(__FAILED__);
  const N = __N__, DRIFT = 0.12;
  const grid = document.getElementById('grid');
  for (let i = 0; i < N; i++) {
    const ok = !FAILED.has(i), nn = String(i).padStart(2,'0');
    const fig = document.createElement('figure');
    fig.innerHTML =
      `<video src="clips/ep${nn}.mp4" muted playsinline preload="auto"
              class="${ok?'pass-b':'fail-b'}"></video>` +
      `<figcaption><span>trial ${i+1} <span style="color:#666">· ep${nn}</span></span>` +
      `<span class="${ok?'pass':'fail'}">${ok?'placed':'failed'}</span></figcaption>`;
    grid.appendChild(fig);
  }
  const vids = [...document.querySelectorAll('video')];
  const seek = document.getElementById('seek'), label = document.getElementById('t');
  function play(){ vids.forEach(v=>v.play().catch(()=>{})); }
  function pauseAll(){ vids.forEach(v=>v.pause()); }
  function seekAll(t){ vids.forEach(v=>{v.currentTime=t;}); label.textContent=t.toFixed(1)+'s'; }
  function restart(){ seekAll(0); play(); }

  // Not the native loop attribute: N elements each looping on their own clock
  // drift apart within a couple of passes. One clip is the master, the rest are
  // corrected to it, and the wrap happens for all of them at once.
  const master = vids[0];
  let DUR = 56.7;
  master.addEventListener('loadedmetadata', () => {
    DUR = Math.min(...vids.map(v => v.duration || 1e9).filter(d => d < 1e9));
    seek.max = DUR;
  });
  setInterval(() => {
    if (master.paused) return;
    const t = master.currentTime;
    for (const v of vids) {
      if (v === master) continue;
      if (Math.abs(v.currentTime - t) > DRIFT) v.currentTime = t;
      if (v.paused) v.play().catch(()=>{});
    }
    seek.value = t; label.textContent = t.toFixed(1)+'s';
  }, 500);
  master.addEventListener('timeupdate', () => { if (master.currentTime >= DUR - 0.05) restart(); });
  master.addEventListener('ended', restart);

  let ready = 0;
  vids.forEach(v => v.addEventListener('canplaythrough', () => { if (++ready === N) restart(); }, {once:true}));
  setTimeout(() => { if (ready < N) restart(); }, 8000);
</script>
"""

if __name__ == "__main__":
    main()
