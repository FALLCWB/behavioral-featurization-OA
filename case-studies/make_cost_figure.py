"""Cost figure: training CPU time per method and domain on a logarithmic scale,
from the 20-repeat resource comparison. Each bar is annotated with the method's test
accuracy. The compact featurization costs one to two orders of magnitude less CPU
than the learned models at comparable accuracy.
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {r["domain"]: r for r in json.load(open(os.path.join(HERE, "results_resource_stats.json")))}

DOMAINS = ["memory", "graphs", "ecology"]
METHODS = [("ours", "Behavioral featurization (ours)", "tab:blue"),
           ("deepsets", "Deep Sets", "tab:gray"),
           ("lstm", "LSTM (order-aware)", "tab:red")]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(DOMAINS))
w = 0.26
for i, (key, label, color) in enumerate(METHODS):
    cpu = [R[d][key]["cpu_s"][0] for d in DOMAINS]
    acc = [R[d][key]["acc"][0] for d in DOMAINS]
    bars = ax.bar(x + (i - 1) * w, cpu, w, color=color, label=label, zorder=3)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{a:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="0.2")

ax.set_yscale("log")
ax.set_ylim(0.1, 1500)
ax.set_xticks(x); ax.set_xticklabels(DOMAINS)
ax.set_ylabel("training CPU time (s, log scale)")
ax.set_xlabel("domain")
ax.set_title("Training cost per method (test accuracy labeled on each bar)")
ax.legend(fontsize=8, loc="upper center", ncol=3, columnspacing=1.0, handletextpad=0.4)
ax.grid(True, axis="y", which="both", ls=":", alpha=0.35, zorder=0)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "F6_cost_accuracy.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved", os.path.relpath(out, HERE))
