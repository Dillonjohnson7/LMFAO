#!/usr/bin/env python
"""Regenerate the live loss curve from the synced train.log (150k run)."""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
txt = open(f"{D}/train.log", errors="replace").read()
# exact step from the progress bar that precedes each INFO loss line
pts = [(int(m.group(1)), float(m.group(2))) for m in
       re.finditer(r"(\d+)/150000 \[[^\]]*\][^\n]*?loss:([0-9.]+)", txt)]
seen = {}
for s, l in pts:
    seen[s] = l
steps = np.array(sorted(seen)); loss = np.array([seen[s] for s in steps])
if len(steps) < 2:
    print("not enough points yet"); sys.exit(0)

BLUE = "#2a78d6"; INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
# prediction from the 20k run's power-law fit
xs = np.linspace(400, 150000, 400)
ax.plot(xs, 624.1 * xs**-0.827, color="#00000030", lw=1.5, ls="--", label="predicted (20k-run fit)")
ax.plot(steps, loss, color=BLUE, lw=1.8, label="measured")
ax.annotate(f"{loss[-1]:.3f} @ {steps[-1]}", (steps[-1], loss[-1]),
            textcoords="offset points", xytext=(8, 8), color=INK, fontsize=9)
ax.set_xlabel("training step", color=INK2); ax.set_ylabel("L1 loss (normalized)", color=INK2)
ax.set_title(f"ACT 150k run — live loss ({len(steps)} points)", color=INK, fontsize=11, loc="left")
ax.set_xlim(0, 152000); ax.set_ylim(0, min(2.3, max(loss) * 1.15))
ax.grid(True, color="#00000012", lw=0.7); ax.tick_params(colors=INK2)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#00000030")
plt.tight_layout(); plt.savefig(f"{D}/loss_curve.png", facecolor=SURF)
print(f"curve: {len(steps)} pts, latest step {steps[-1]} loss {loss[-1]:.3f}")
