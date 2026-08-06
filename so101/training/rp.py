"""RunPod GraphQL driver: deploy / status / list / waitssh / terminate a GPU pod.

Preserved here (2026-07-18) from an ephemeral /tmp scratchpad that a reboot would
erase — it is the reusable driver behind every cloud retrain in NEXT_STEPS.md
(options 1/2/5/7), not a one-off. No secret lives in this file: the API key comes
from $RUNPOD_API_KEY, or a gitignored `rp_key` file next to this script. The
Cloudflare edge 1010-blocks a default urllib UA, hence the curl User-Agent.

  RUNPOD_API_KEY=... python training/rp.py list
  python training/rp.py deploy [COMMUNITY|SECURE] ["<gpu name>"] [image]
                # default: SECURE "NVIDIA RTX A5000" — the proven $0.27/hr
                # 150k-run pattern; community boxes can be duds (Mistake #4)
  python training/rp.py waitssh <pod-id>      # prints SSH_READY ip=... port=...
  python training/rp.py terminate <pod-id>
"""
import json, os, sys, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_key():
    key = os.getenv("RUNPOD_API_KEY")
    if key:
        return key.strip()
    try:
        return open(os.path.join(HERE, "rp_key")).read().strip()
    except OSError:
        sys.exit("no RunPod key: set RUNPOD_API_KEY or put it in training/rp_key (gitignored)")


KEY = _load_key()


def gql(query):
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={KEY}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"HTTPError": e.code, "body": e.read().decode()}


cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "deploy":
    cloud = sys.argv[2] if len(sys.argv) > 2 else "SECURE"
    gpu = sys.argv[3] if len(sys.argv) > 3 else "NVIDIA RTX A5000"
    img = sys.argv[4] if len(sys.argv) > 4 else "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    pub = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    env = f'[{{key:"PUBLIC_KEY", value:{json.dumps(pub)}}}]'
    mut = ('mutation { podFindAndDeployOnDemand(input: {'
           ' cloudType: ' + cloud + ', gpuCount: 1, gpuTypeId: "' + gpu + '",'
           ' name: "so101-act-train", imageName: "' + img + '",'
           ' containerDiskInGb: 40, volumeInGb: 0, ports: "22/tcp",'
           ' env: ' + env + ' }) { id desiredStatus imageName machineId } }')
    print(json.dumps(gql(mut), indent=2))
elif cmd == "status":
    pid = sys.argv[2]
    q = ('query { pod(input:{podId:"' + pid + '"}) { id name desiredStatus '
         'runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }')
    print(json.dumps(gql(q), indent=2))
elif cmd == "list":
    q = ('query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus '
         'costPerHr gpuCount machine { gpuDisplayName } runtime { uptimeInSeconds } } } }')
    print(json.dumps(gql(q), indent=2))
elif cmd == "waitssh":
    import time
    pid = sys.argv[2]
    last = None
    for _ in range(72):  # up to ~6 min
        q = ('query { pod(input:{podId:"' + pid + '"}) { desiredStatus runtime {'
             ' uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }')
        last = gql(q)
        pod = (last.get("data") or {}).get("pod") or {}
        for p in (pod.get("runtime") or {}).get("ports") or []:
            if p.get("privatePort") == 22 and p.get("type") == "tcp" and p.get("isIpPublic"):
                print(f"SSH_READY ip={p['ip']} port={p['publicPort']}")
                sys.exit(0)
        time.sleep(5)
    print("TIMEOUT; last=" + json.dumps(last))
    sys.exit(1)
elif cmd == "terminate":
    pid = sys.argv[2]
    print(json.dumps(gql('mutation { podTerminate(input:{podId:"' + pid + '"}) }'), indent=2))
else:
    sys.exit(f"usage: python training/rp.py [deploy|status|list|waitssh|terminate] ...  (got {cmd!r})")
