import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "summary.json")))
rows = [r for r in d["rows"] if not r["policy"].startswith("ours_minus")]
colors = {"b0": "#d95f02", "b1": "#7570b3", "b2": "#1b9e77", "ours": "#000000"}
labels = {"b0": "B0 trust first", "b1": "B1 tuned threshold", "b2": "B2 always verify", "ours": "ours (DP policy)"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for pol in ("b0", "b1", "b2", "ours"):
    rs = sorted([r for r in rows if r["policy"] == pol and r["mean_ttc"] == r["mean_ttc"]], key=lambda r: r["fd"])
    ax[0].errorbar([r["wasted"] for r in rs], [r["mean_ttc"] for r in rs],
                   xerr=[[r["wasted"] - r["wasted_lo"] for r in rs], [r["wasted_hi"] - r["wasted"] for r in rs]],
                   yerr=[[r["mean_ttc"] - r["mean_ttc_lo"] for r in rs], [r["mean_ttc_hi"] - r["mean_ttc"] for r in rs]],
                   marker="o", ls="-" if pol != "ours" else "--", color=colors[pol], label=labels[pol], capsize=2, alpha=0.9)
    rc = sorted([r for r in rows if r["policy"] == pol], key=lambda r: r["fd"])
    ax[1].errorbar([r["fd"] for r in rc], [r["cost"] for r in rc],
                   yerr=[[r["cost"] - r["cost_lo"] for r in rc], [r["cost_hi"] - r["cost"] for r in rc]],
                   marker="o", ls="-" if pol != "ours" else "--", color=colors[pol], label=labels[pol], capsize=2)
ax[0].set_xlabel("wasted dispatches per mission")
ax[0].set_ylabel("mean time to confirm a victim (s)")
ax[0].set_title("Time to confirm vs wasted dispatches\n(each point = one false-dispatch cost, 10 to 320)")
ax[1].set_xscale("log")
ax[1].set_xlabel("false-dispatch cost")
ax[1].set_ylabel("mean realised cost per mission")
ax[1].set_title("Realised cost vs false-dispatch cost")
ax[0].legend()
for a in ax:
    a.grid(alpha=0.3)
fig.suptitle(f"200 seeds, 95% bootstrap intervals; fast simulator calibrated to Webots", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "tradeoff.png"), dpi=140)
