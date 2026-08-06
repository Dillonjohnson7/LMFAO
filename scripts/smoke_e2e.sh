#!/usr/bin/env bash
# End-to-end smoke: inspect stock demos → tiny augment buckets → reload outputs.
#
# Usage:
#   STOCK=/path/to/lerobot_dataset ./scripts/smoke_e2e.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOCK="${STOCK:?set STOCK to a LeRobot dataset root (with meta/)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/smoke_e2e}"
VARIANTS="${VARIANTS:-1}"
SEED="${SEED:-7}"
BUCKETS="${BUCKETS:-lighting noise occlusion spatial full}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/training/buckets}"

if [[ ! -d "$STOCK/meta" ]]; then
  echo "STOCK does not look like a LeRobot dataset: ${STOCK}" >&2
  exit 1
fi

cd "$ROOT"
# Prefer repo venv if present
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
if ! command -v lmfao >/dev/null 2>&1; then
  pip install -e ".[lerobot]" >/dev/null
fi

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

echo "=== 1) inspect stock ==="
lmfao inspect "$STOCK" --episodes
python - <<PY
from pathlib import Path
import json
root = Path("$STOCK")
info = json.loads((root / "meta" / "info.json").read_text())
assert info.get("codebase_version") == "v3.0", info.get("codebase_version")
assert info.get("total_episodes", 0) >= 1
print(f"stock OK: v3.0 · {info['total_episodes']} eps · {info['total_frames']} frames · fps={info['fps']}")
PY

echo
echo "=== 2) augment smoke buckets (variants=${VARIANTS}) ==="
STOCK_LINK="${OUT_ROOT}/stock"
ln -s "$(cd "$STOCK" && pwd)" "$STOCK_LINK"

for bucket in $BUCKETS; do
  cfg="${CONFIG_DIR}/${bucket}.json"
  out="${OUT_ROOT}/${bucket}"
  [[ -f "$cfg" ]] || { echo "missing ${cfg}" >&2; exit 1; }
  echo "--- bucket=${bucket} ---"
  lmfao augment \
    --input "$STOCK" \
    --output "$out" \
    --config "$cfg" \
    --variants "$VARIANTS" \
    --include-original \
    --seed "$SEED"
  lmfao inspect "$out" --episodes | head -n 30
done

echo
echo "=== 3) reload / schema checks ==="
OUT_ROOT="$OUT_ROOT" STOCK="$STOCK" BUCKETS="$BUCKETS" VARIANTS="$VARIANTS" python - <<'PY'
from pathlib import Path
import json
import os
import pyarrow.parquet as pq

out_root = Path(os.environ["OUT_ROOT"])
stock = Path(os.environ["STOCK"])
buckets = os.environ["BUCKETS"].split()
variants = int(os.environ["VARIANTS"])

info_in = json.loads((stock / "meta" / "info.json").read_text())
n_in = int(info_in["total_episodes"])
# include-original => originals + variants per source episode
expect_eps = n_in * (1 + variants)

img_keys = sorted(k for k in info_in["features"] if k.startswith("observation.images."))
assert img_keys, "stock has no image features"
# Dual-cam SO101 must not silently drop wrist/front.
for required in ("observation.images.front", "observation.images.wrist"):
    if required in info_in["features"] and required not in img_keys:
        raise SystemExit(f"stock missing expected camera {required}")

errors = []
for bucket in buckets:
    root = out_root / bucket
    info_p = root / "meta" / "info.json"
    if not info_p.exists():
        errors.append(f"{bucket}: missing meta/info.json")
        continue
    info = json.loads(info_p.read_text())
    if info.get("codebase_version") != "v3.0":
        errors.append(f"{bucket}: bad codebase_version {info.get('codebase_version')}")
    got = int(info.get("total_episodes", 0))
    if got != expect_eps:
        errors.append(f"{bucket}: episodes {got} != expected {expect_eps}")
    # features preserved — every stock camera must survive
    for k in img_keys + ["action", "observation.state"]:
        if k not in info.get("features", {}):
            errors.append(f"{bucket}: missing feature {k}")
    # parquet readable
    parts = list((root / "data").rglob("*.parquet"))
    if not parts:
        errors.append(f"{bucket}: no data parquet")
    else:
        n_rows = sum(pq.read_table(p).num_rows for p in parts)
        if n_rows != int(info.get("total_frames", -1)):
            errors.append(f"{bucket}: parquet rows {n_rows} != total_frames {info.get('total_frames')}")
    # videos for each image key (count must match episode count)
    for k in img_keys:
        vids = list((root / "videos" / k).rglob("*.mp4"))
        if not vids:
            errors.append(f"{bucket}: no videos for {k}")
        elif len(vids) != got:
            errors.append(f"{bucket}: {k} has {len(vids)} mp4s != {got} episodes")
    # provenance sidecar if present
    prov = root / "meta" / "lmfao_provenance.json"
    if not prov.exists():
        errors.append(f"{bucket}: missing meta/lmfao_provenance.json")
    kept = sorted(k for k in info.get("features", {}) if k.startswith("observation.images."))
    print(f"{bucket}: eps={got} frames={info.get('total_frames')} imgs={kept}")

if errors:
    print("SMOKE FAIL:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print(f"SMOKE_OK · stock={n_in} eps · buckets={buckets} · expect {expect_eps} eps each")
PY

echo
echo "Smoke outputs: ${OUT_ROOT}"
