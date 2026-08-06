#!/usr/bin/env python3
"""Live training dashboard for the 150k run — read-only, zero impact on training.

ARCHIVE / NON-RERUNNABLE: this is a point-in-time record of the completed 150k
run. It reads pod_ssh.txt plus a baseline_balance.txt / rp_key from an ephemeral
/tmp scratchpad that no longer exists, and its pod is long terminated — so it
will not run as-is. Kept for the curves and the parsing logic. The REUSABLE
RunPod driver was preserved at training/rp.py (../rp.py); point future runs there.

Syncs train.log off the pod every 45 s (scp of one file), parses it, and serves
a self-refreshing page: game-style HP/progress bar + big live loss curve.
  http://localhost:8095      (workcell)
  <your-tailscale-ip>:8095   (Tailscale, e.g. phone)
"""
import http.server, json, re, subprocess, threading, time, os

D = os.path.dirname(os.path.abspath(__file__))
SM = "/tmp/scratchpad"
IP, PORT_SSH = open(f"{D}/pod_ssh.txt").read().split()
TOTAL = 150_000
START_BALANCE = float(open(f"{SM}/baseline_balance.txt").read().strip())
RP_KEY = open(f"{SM}/rp_key").read().strip()

def real_spend():
    """REAL spend this run: baseline balance minus live balance, from RunPod's API."""
    import urllib.request
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={RP_KEY}",
        data=json.dumps({"query": "query{myself{clientBalance}}"}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    bal = json.load(urllib.request.urlopen(req, timeout=15))["data"]["myself"]["clientBalance"]
    return round(START_BALANCE - bal, 3), round(bal, 2)

STATE = {"pts": [], "step": 0, "loss": None, "sps": None, "eta_s": None,
         "ckpts": 0, "gpu": "", "synced": None, "spend": None, "balance": None, "done": False}

SSH_OPTS = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            "-o ConnectTimeout=10 -o BatchMode=yes -i $HOME/.ssh/id_ed25519")

def sync_loop():
    t0 = time.time()
    while True:
        try:
            subprocess.run(f"scp {SSH_OPTS} -P {PORT_SSH} root@{IP}:/root/train.log {D}/train.log.tmp",
                           shell=True, capture_output=True, timeout=40)
            if os.path.getsize(f"{D}/train.log.tmp") > 0:
                os.replace(f"{D}/train.log.tmp", f"{D}/train.log")
            subprocess.run(f"scp {SSH_OPTS} -P {PORT_SSH} root@{IP}:/root/gpu.log {D}/gpu.log",
                           shell=True, capture_output=True, timeout=30)
            subprocess.run(f"scp {SSH_OPTS} -P {PORT_SSH} root@{IP}:/root/train.done {D}/train.done",
                           shell=True, capture_output=True, timeout=20)
        except Exception:
            pass
        try:
            txt = open(f"{D}/train.log", errors="replace").read()
            pts = {}
            for m in re.finditer(r"(\d+)/150000 \[[^\]]*?([0-9.]+)step/s[^\n]*?loss:([0-9.]+)", txt):
                pts[int(m.group(1))] = (float(m.group(3)), float(m.group(2)))
            # REAL numbers only: tqdm's own remaining-time estimate, not our arithmetic
            bars = re.findall(r"(\d+)/150000 \[([\d:]+)<([\d:]+),\s*([0-9.]+)step/s", txt[-4000:])
            step = max((int(b[0]) for b in bars), default=max(pts, default=0))
            sps = float(bars[-1][3]) if bars else None
            eta_s = elapsed_s = None
            if bars:
                def hms(t):
                    p = [int(x) for x in t.split(":")]
                    return sum(v * m for v, m in zip(reversed(p), (1, 60, 3600)))
                elapsed_s, eta_s = hms(bars[-1][1]), hms(bars[-1][2])
            STATE["pts"] = [[s, pts[s][0]] for s in sorted(pts)]
            STATE["step"] = step
            STATE["loss"] = STATE["pts"][-1][1] if STATE["pts"] else None
            STATE["sps"] = sps
            STATE["eta_s"] = eta_s
            STATE["elapsed_s"] = elapsed_s
            try:
                STATE["spend"], STATE["balance"] = real_spend()   # RunPod billing API
            except Exception:
                pass
            # live checkpoint count on the POD itself (not the lagging local mirror)
            try:
                r = subprocess.run(
                    f"ssh {SSH_OPTS} -p {PORT_SSH} root@{IP} "
                    "'ls /root/outputs/train/act_pick_place/checkpoints 2>/dev/null | grep -c \"^[0-9]\"'",
                    shell=True, capture_output=True, timeout=25, text=True)
                STATE["ckpts"] = int(r.stdout.strip() or 0)
            except Exception:
                pass
            g = open(f"{D}/gpu.log", errors="replace").read().strip().splitlines()
            STATE["gpu"] = g[-1] if g else ""
            STATE["done"] = os.path.exists(f"{D}/train.done") and os.path.getsize(f"{D}/train.done") > 0
            STATE["synced"] = time.strftime("%H:%M:%S")
        except Exception:
            pass
        time.sleep(45)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO101 · 150k run</title>
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
 .done{color:#7fd08c}
</style></head><body><div class="wrap">
<h1>SO101 · ACT 150k TRAINING RUN <span id="donebadge"></span></h1>
<div class="hpwrap"><div class="hp" id="hp">
  <div class="hpfill" id="hpfill" style="width:0%"></div><div class="hptext" id="hptext"></div>
</div></div>
<div class="stats">
 <div class="tile"><b id="loss">–</b><span>loss</span></div>
 <div class="tile"><b id="sps">–</b><span>steps / sec</span></div>
 <div class="tile"><b id="elapsed">–</b><span>elapsed (tqdm)</span></div>
 <div class="tile"><b id="eta">–</b><span>eta (tqdm)</span></div>
 <div class="tile"><b id="spend">–</b><span>spent · runpod api</span></div>
 <div class="tile"><b id="gpu">–</b><span>gpu util (5-min sample)</span></div>
</div>
<div class="chartbox"><canvas id="c" width="2100" height="920"></canvas></div>
<div class="foot">auto-refreshes every 30 s · synced <span id="sync">–</span> · read-only (training untouched) · predicted curve: 624·step<sup>-0.83</sup></div>
</div><script>
const cv=document.getElementById('c'),cx=cv.getContext('2d');
function fmts(s){if(s==null)return'–';const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h+'h '+String(m).padStart(2,'0')+'m'}
function draw(d){
 const W=cv.width,H=cv.height,L=90,R=30,T=30,B=60;
 cx.clearRect(0,0,W,H);
 const xmax=150000, ymax=Math.min(2.4,(d.pts.length?Math.max(...d.pts.map(p=>p[1])):2.4)*1.12);
 const X=s=>L+(W-L-R)*s/xmax, Y=v=>T+(H-T-B)*(1-v/ymax);
 cx.strokeStyle='#ffffff14';cx.fillStyle='#8f8e8a';cx.font='24px sans-serif';cx.lineWidth=1;
 for(let v=0;v<=ymax;v+=0.5){cx.beginPath();cx.moveTo(L,Y(v));cx.lineTo(W-R,Y(v));cx.stroke();cx.fillText(v.toFixed(1),18,Y(v)+8)}
 for(let s=0;s<=xmax;s+=25000){cx.beginPath();cx.moveTo(X(s),T);cx.lineTo(X(s),H-B);cx.stroke();cx.fillText((s/1000)+'k',X(s)-16,H-B+34)}
 cx.strokeStyle='#ffffff3a';cx.setLineDash([8,7]);cx.lineWidth=2.5;cx.beginPath();
 for(let s=600;s<=xmax;s+=800){const v=624.1*Math.pow(s,-0.827);const y=Y(Math.min(v,ymax));s===600?cx.moveTo(X(s),y):cx.lineTo(X(s),y)}
 cx.stroke();cx.setLineDash([]);
 if(d.pts.length>1){cx.strokeStyle='#4b9be8';cx.lineWidth=4;cx.shadowColor='#2a78d6aa';cx.shadowBlur=10;cx.beginPath();
  d.pts.forEach((p,i)=>{i?cx.lineTo(X(p[0]),Y(p[1])):cx.moveTo(X(p[0]),Y(p[1]))});cx.stroke();cx.shadowBlur=0;
  const lp=d.pts[d.pts.length-1];cx.fillStyle='#4b9be8';cx.beginPath();cx.arc(X(lp[0]),Y(lp[1]),8,0,7);cx.fill();
  cx.fillStyle='#fcfcfb';cx.font='26px sans-serif';cx.fillText(lp[1].toFixed(3),X(lp[0])+14,Y(lp[1])-10)}
}
async function tick(){
 try{
  const d=await (await fetch('/data.json')).json();
  const pct=100*d.step/150000;
  document.getElementById('hpfill').style.width=pct+'%';
  document.getElementById('hptext').textContent=d.step.toLocaleString()+' / 150,000  ('+pct.toFixed(1)+'%)';
  const hp=document.getElementById('hp');
  [...hp.querySelectorAll('.seg')].forEach(e=>e.remove());
  for(let i=1;i<15;i++){const s=document.createElement('div');s.className='seg';s.style.left=(100*i/15)+'%';hp.appendChild(s)}
  document.getElementById('loss').textContent=d.loss!=null?d.loss.toFixed(3):'–';
  document.getElementById('sps').textContent=d.sps!=null?d.sps.toFixed(2):'–';
  document.getElementById('elapsed').textContent=fmts(d.elapsed_s);
  document.getElementById('eta').textContent=d.done?'DONE':fmts(d.eta_s);
  document.getElementById('spend').textContent=d.spend!=null?('$'+d.spend.toFixed(2)+' (bal $'+d.balance.toFixed(2)+')'):'–';
  document.getElementById('gpu').textContent=d.gpu?d.gpu.split(',')[0]:'–';
  document.getElementById('sync').textContent=d.synced||'–';
  document.getElementById('donebadge').innerHTML=d.done?'<span class="done">· ✔ COMPLETE</span>':'';
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
    print("dashboard on http://0.0.0.0:8095")
    http.server.ThreadingHTTPServer(("0.0.0.0", 8095), H).serve_forever()
