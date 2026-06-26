"""One viewable dashboard: statistical results table + contrast bar chart."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
stats = json.load(open(os.path.join(HERE, "results_stats.json")))
contrast = json.load(open(os.path.join(HERE, "results_contrast_ablation.json")))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

# --- table: statistical results ---
rows = []
for s in stats:
    rows.append([
        s["name"].split(":")[0],
        f"{s['acc_mean']:.3f}",
        f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]",
        f"{s['chance']:.3f}",
        f"{s['perm_p']:.3f}",
        ">98%" if s["sig_98"] else (">95%" if s["sig_95"] else "n.s."),
    ])
ax1.axis("off")
tbl = ax1.table(cellText=rows,
                colLabels=["Domain / test", "Accuracy", "95% CI", "Chance", "perm p", "Signif."],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for j in range(6):
    tbl[0, j].set_facecolor("#34495e")
    tbl[0, j].set_text_props(color="white", weight="bold")
ax1.set_title("Statistical results: repeated 5-fold CV (20 repeats) + label-permutation test (200 perms)",
              fontsize=12, weight="bold", pad=14)

# --- bars: contrast with the giants ---
methods = ["Behavioral Featurization (ours)", "Set-pool (Deep Sets-style)",
           "Sketch (synopsis)", "Naive (final snapshot)"]
short = ["Behavioral (ours)", "Set-pool (Deep Sets)", "Sketch", "Naive snapshot"]
colors = ["#2980b9", "#27ae60", "#8e44ad", "#c0392b"]
domains = [c["domain"] for c in contrast]
x = np.arange(len(domains))
w = 0.2
for i, m in enumerate(methods):
    vals = [c["contrast"][m][0] for c in contrast]
    errs = [c["contrast"][m][1] for c in contrast]
    ax2.bar(x + i * w, vals, w, yerr=errs, capsize=3, label=short[i], color=colors[i])
ax2.axhline(1 / 3, ls="--", color="gray", lw=1, label="chance (0.333)")
ax2.set_xticks(x + 1.5 * w)
ax2.set_xticklabels(domains)
ax2.set_ylabel("CV accuracy")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, ncol=2)
ax2.set_title("Contrast with the giants: naive snapshot collapses; pool/sketch tie (they are instances)",
              fontsize=12, weight="bold")
for i, c in enumerate(contrast):
    ax2.text(x[i] + 1.5 * w, 0.02, f"chance {c['chance']:.2f}", ha="center", fontsize=8, color="gray")

fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "SUMMARY_dashboard.png")
fig.savefig(out, dpi=140)
print("saved:", os.path.relpath(out, HERE))
