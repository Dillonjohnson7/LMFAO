#!/usr/bin/env python3
"""Live training dashboard for the V2 (wrist-cam) run — read-only, zero impact.

Rebuilt 2026-07-21 from the archived run150k/dashboard.py, with every ephemeral
dependency removed (Mistake: the v1 copy read keys/baselines from a /tmp
scratchpad that a reboot erased). Everything lives in the repo now:
  key       : training/rp_key            (gitignored; same file rp.py uses)
  pod       : training/run_v2/pod_ssh.txt          (written by launch.sh)
  baseline  : training/run_v2/baseline_balance.txt (written by launch.sh)
Starts fine BEFORE any pod exists — shows WAITING FOR POD and the v1 reference
curve; goes live the moment launch.sh writes pod_ssh.txt.

Parsing is unit-agnostic: tqdm reports "step/s" on GPU but flips to "s/step"
below 1 step/s (seen on the v2mini CPU run) — both are handled, as is the
"[00:00<?, ?step/s]" warmup form. Step total is parsed from the log itself
(denominator of the tqdm bar), so STEPS overrides need no dashboard change.

  http://localhost:8095      (workcell)
  <your-tailscale-ip>:8095   (Tailscale, e.g. phone)

Env: DASH_DIR (default: this file's dir) — fixture override for tests.
     DASH_PORT (default 8095) · TOTAL (default 120000, until the log knows better)
"""
import http.server, json, os, re, subprocess, threading, time

D = os.environ.get("DASH_DIR", os.path.dirname(os.path.abspath(__file__)))
REPO_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # training/
PORT = int(os.environ.get("DASH_PORT", "8095"))
TOTAL_FALLBACK = int(os.environ.get("TOTAL", "120000"))
JOB = os.environ.get("JOB", "act_pick_place_v2")

SSH_OPTS = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            "-o ConnectTimeout=10 -o BatchMode=yes -i $HOME/.ssh/id_ed25519")

STATE = {"pts": [], "step": 0, "total": TOTAL_FALLBACK, "loss": None, "sps": None,
         "eta_s": None, "elapsed_s": None, "ckpts": 0, "gpu": "", "synced": None,
         "spend": None, "balance": None, "done": False, "waiting": True, "ref": []}


def pod():
    """(ip, port) or None — the pod may not exist yet; that is a valid state."""
    try:
        ip, p = open(f"{D}/pod_ssh.txt").read().split()
        return ip, p
    except Exception:
        return None


def real_spend():
    """REAL spend this run: baseline balance minus live balance (RunPod API)."""
    import urllib.request
    key = open(os.path.join(REPO_TRAIN, "rp_key")).read().strip()
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": "query{myself{clientBalance}}"}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    bal = json.load(urllib.request.urlopen(req, timeout=15))["data"]["myself"]["clientBalance"]
    b0 = float(open(f"{D}/baseline_balance.txt").read().strip())
    return round(b0 - bal, 3), round(bal, 2)


def hms(t):
    if "?" in t:
        return None
    p = [int(x) for x in t.split(":")]
    return sum(v * m for v, m in zip(reversed(p), (1, 60, 3600)))


# tqdm fragment: "12345/120000 [1:23:45<2:34:56,  3.21step/s"  OR  "...  7.5s/step"
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([0-9.?]+)\s*(step/s|s/step)")
# curve point: tqdm fragment right before an INFO line that carries loss:
PT = re.compile(r"(\d+)/(\d+) \[[^\]]*\][^\n]*?loss:([0-9.]+)")


def parse_log(txt):
    """-> (pts, step, total, sps, elapsed_s, eta_s) — None-safe on partial logs."""
    pts = {}
    total = TOTAL_FALLBACK
    for m in PT.finditer(txt):
        pts[int(m.group(1))] = float(m.group(3))
        total = int(m.group(2))
    bars = BAR.findall(txt[-4000:])
    step = max((int(b[0]) for b in bars), default=max(pts, default=0))
    if bars:
        total = int(bars[-1][1])
    sps = elapsed_s = eta_s = None
    if bars:
        rate, unit = bars[-1][4], bars[-1][5]
        if "?" not in rate:
            sps = float(rate) if unit == "step/s" else (1.0 / float(rate) if float(rate) else None)
        elapsed_s, eta_s = hms(bars[-1][2]), hms(bars[-1][3])
    return [[s, pts[s]] for s in sorted(pts)], step, total, sps, elapsed_s, eta_s


def load_reference():
    """The v1 150k run's ACTUAL curve (downsampled) as a grey reference — real
    numbers, not a fit: the power-law extrapolation missed the floor by 75%."""
    try:
        txt = open(f"{REPO_TRAIN}/run150k/train.log", errors="replace").read()
        pts = [[int(m.group(1)), float(m.group(3))] for m in PT.finditer(txt)]
        return pts[:: max(1, len(pts) // 150)]
    except Exception:
        return []


def sync_loop():
    STATE["ref"] = load_reference()
    while True:
        p = pod()
        STATE["waiting"] = p is None
        if p:
            ip, port_ssh = p
            for f in ("train.log", "gpu.log", "train.done"):
                subprocess.run(f"scp {SSH_OPTS} -P {port_ssh} root@{ip}:/root/{f} {D}/{f}.tmp",
                               shell=True, capture_output=True, timeout=60)
                try:
                    if os.path.getsize(f"{D}/{f}.tmp") > 0 or f == "train.done":
                        os.replace(f"{D}/{f}.tmp", f"{D}/{f}")
                except OSError:
                    pass
            try:  # live checkpoint count on the pod (not the lagging mirror)
                r = subprocess.run(
                    f"ssh {SSH_OPTS} -p {port_ssh} root@{ip} "
                    f"'ls /root/outputs/train/{JOB}/checkpoints 2>/dev/null | grep -c \"^[0-9]\"'",
                    shell=True, capture_output=True, timeout=25, text=True)
                STATE["ckpts"] = int(r.stdout.strip() or 0)
            except Exception:
                pass
        try:
            txt = open(f"{D}/train.log", errors="replace").read()
            STATE["pts"], STATE["step"], STATE["total"], STATE["sps"], \
                STATE["elapsed_s"], STATE["eta_s"] = parse_log(txt)
            STATE["loss"] = STATE["pts"][-1][1] if STATE["pts"] else None
        except FileNotFoundError:
            pass
        except Exception:
            pass
        try:
            STATE["spend"], STATE["balance"] = real_spend()
        except Exception:
            pass
        try:
            g = open(f"{D}/gpu.log", errors="replace").read().strip().splitlines()
            STATE["gpu"] = g[-1] if g else ""
        except Exception:
            pass
        try:
            STATE["done"] = os.path.getsize(f"{D}/train.done") > 0
        except OSError:
            STATE["done"] = False
        STATE["synced"] = time.strftime("%H:%M:%S")
        time.sleep(45)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO101 · v2 wrist-cam run</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#141413;color:#fcfcfb;font:14px/1.5 -apple-system,system-ui,sans-serif}
 .wrap{max-width:1100px;margin:0 auto;padding:22px 18px}
 h1{font-size:15px;font-weight:600;color:#a3a29e;margin:0 0 14px;letter-spacing:.4px}
 .hpwrap{background:#232322;border:1px solid #33332f;border-radius:10px;padding:5px;box-shadow:0 0 24px #0006}
 .hp{height:42px;border-radius:6px;position:relative;overflow:hidden;background:#1b1b1a}
 .hpfill{height:100%;border-radius:6px;background:linear-gradient(90deg,#2a78d6,#4b9be8);
   box-shadow:0 0 14px #2a78d688;transition:width 1s ease}
 .hp .seg{position:absolute;top:0;bottom:0;width:1px;background:#14141366}
 .hptext{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   font-weight:700;font-size:17px;text-shadow:0 1px 3px #000c;letter-spacing:.5px}
 .stats{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 20px}
 .tile{background:#1d1d1c;border:1px solid #2b2b29;border-radius:8px;padding:10px 16px;min-width:118px}
 .tile b{display:block;font-size:20px;font-weight:650;color:#fcfcfb}
 .tile span{color:#8f8e8a;font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
 .chartbox{background:#1a1a19;border:1px solid #2b2b29;border-radius:10px;padding:14px}
 canvas{width:100%;height:460px;display:block}
 .foot{color:#6f6e6a;font-size:12px;margin-top:10px}
 .done{color:#7fd08c}.wait{color:#e8b84b}
</style></head><body><div class="wrap">
<h1>SO101 · ACT V2 (WRIST-CAM) TRAINING <span id="badge"></span></h1>
<div class="hpwrap"><div class="hp" id="hp">
  <div class="hpfill" id="hpfill" style="width:0%"></div><div class="hptext" id="hptext"></div>
</div></div>
<div class="stats">
 <div class="tile"><b id="loss">–</b><span>loss</span></div>
 <div class="tile"><b id="sps">–</b><span>steps / sec</span></div>
 <div class="tile"><b id="elapsed">–</b><span>elapsed (tqdm)</span></div>
 <div class="tile"><b id="eta">–</b><span>eta (tqdm)</span></div>
 <div class="tile"><b id="spend">–</b><span>spent · runpod api</span></div>
 <div class="tile"><b id="ckpts">–</b><span>checkpoints</span></div>
 <div class="tile"><b id="gpu">–</b><span>gpu util (5-min sample)</span></div>
</div>
<div class="chartbox"><canvas id="c" width="2100" height="920"></canvas></div>
<div class="foot">auto-refreshes every 30 s · synced <span id="sync">–</span> · read-only (training untouched)
 · grey reference: the v1 150k run's ACTUAL curve (1 front cam, 30 demos) · y-axis clipped to late-run range</div>
</div><script>
const cv=document.getElementById('c'),cx=cv.getContext('2d');
function fmts(s){if(s==null)return'–';const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h+'h '+String(m).padStart(2,'0')+'m'}
function draw(d){
 const W=cv.width,H=cv.height,L=90,R=30,T=30,B=60;
 cx.clearRect(0,0,W,H);
 const xmax=d.total||120000;
 const late=d.pts.filter(p=>p[0]>2000).map(p=>p[1]);
 const ymax=Math.min(4,(late.length?Math.max(...late):(d.pts.length?Math.max(...d.pts.map(p=>p[1])):2.4))*1.15)||2.4;
 const X=s=>L+(W-L-R)*Math.min(s,xmax)/xmax, Y=v=>T+(H-T-B)*(1-Math.min(v,ymax)/ymax);
 cx.strokeStyle='#ffffff14';cx.fillStyle='#8f8e8a';cx.font='24px sans-serif';cx.lineWidth=1;
 const ystep=ymax>2?0.5:0.25;
 for(let v=0;v<=ymax;v+=ystep){cx.beginPath();cx.moveTo(L,Y(v));cx.lineTo(W-R,Y(v));cx.stroke();cx.fillText(v.toFixed(2),8,Y(v)+8)}
 const xstep=xmax>=100000?20000:Math.max(1,Math.round(xmax/6/1000))*1000;
 for(let s=0;s<=xmax;s+=xstep){cx.beginPath();cx.moveTo(X(s),T);cx.lineTo(X(s),H-B);cx.stroke();cx.fillText((s/1000)+'k',X(s)-16,H-B+34)}
 if(d.ref&&d.ref.length>1){cx.strokeStyle='#ffffff30';cx.lineWidth=2.5;cx.beginPath();let started=false;
  d.ref.forEach(p=>{if(p[0]>xmax)return;const x=X(p[0]),y=Y(p[1]);started?cx.lineTo(x,y):cx.moveTo(x,y);started=true});cx.stroke()}
 if(d.pts.length>1){cx.strokeStyle='#4b9be8';cx.lineWidth=4;cx.shadowColor='#2a78d6aa';cx.shadowBlur=10;cx.beginPath();
  d.pts.forEach((p,i)=>{i?cx.lineTo(X(p[0]),Y(p[1])):cx.moveTo(X(p[0]),Y(p[1]))});cx.stroke();cx.shadowBlur=0;
  const lp=d.pts[d.pts.length-1];cx.fillStyle='#4b9be8';cx.beginPath();cx.arc(X(lp[0]),Y(lp[1]),8,0,7);cx.fill();
  cx.fillStyle='#fcfcfb';cx.font='26px sans-serif';cx.fillText(lp[1].toFixed(3),Math.min(X(lp[0])+14,W-140),Y(lp[1])-10)}
}
async function tick(){
 try{
  const d=await (await fetch('/data.json')).json();
  const pct=100*d.step/(d.total||120000);
  document.getElementById('hpfill').style.width=pct+'%';
  document.getElementById('hptext').textContent=d.step.toLocaleString()+' / '+(d.total||120000).toLocaleString()+'  ('+pct.toFixed(1)+'%)';
  const hp=document.getElementById('hp');
  [...hp.querySelectorAll('.seg')].forEach(e=>e.remove());
  for(let i=1;i<12;i++){const s=document.createElement('div');s.className='seg';s.style.left=(100*i/12)+'%';hp.appendChild(s)}
  document.getElementById('loss').textContent=d.loss!=null?d.loss.toFixed(3):'–';
  document.getElementById('sps').textContent=d.sps!=null?d.sps.toFixed(2):'–';
  document.getElementById('elapsed').textContent=fmts(d.elapsed_s);
  document.getElementById('eta').textContent=d.done?'DONE':fmts(d.eta_s);
  document.getElementById('spend').textContent=d.spend!=null?('$'+d.spend.toFixed(2)+' (bal $'+d.balance.toFixed(2)+')'):'–';
  document.getElementById('ckpts').textContent=d.ckpts||'–';
  document.getElementById('gpu').textContent=d.gpu?d.gpu.split(',')[0]:'–';
  document.getElementById('sync').textContent=d.synced||'–';
  document.getElementById('badge').innerHTML=d.done?'<span class="done">· ✔ COMPLETE</span>'
    :(d.waiting?'<span class="wait">· ⏳ WAITING FOR POD (launch.sh not run yet)</span>':'');
  draw(d);
 }catch(e){}
}
tick();setInterval(tick,30000);
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body = json.dumps(STATE).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
        else:
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    threading.Thread(target=sync_loop, daemon=True).start()
    print(f"dashboard on http://0.0.0.0:{PORT}")
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
