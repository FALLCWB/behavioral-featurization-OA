"""Window-size sweep across all three domains: how source-classification accuracy
depends on the observation window length, the principle's one free parameter. Each
domain is swept from a quarter to twice its default window and plotted against the
relative window length (window / default), so the heterogeneous window units (time
steps for memory, edges for graphs, samples for ecology) share one axis. Same
leakage-controlled cross-validation as the main results.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "ecology-ecomon", "graphs-dynamic"]:
    sys.path.insert(0, os.path.join(HERE, d))
import real_memory_demo as MEM
import ecology_demo as ECO
import graphs_demo as GR
from pipeline import featurize_signals

SEED = 0
REL = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def ci95(a):
    a = np.asarray(a, float)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(h)


def cv_acc(sigs, y, blocks):
    X = np.array([featurize_signals(s) for s in sigs])
    if blocks is None:
        cv, groups = StratifiedKFold(5, shuffle=True, random_state=SEED), None
    else:
        cv, groups = StratifiedGroupKFold(5), blocks
    sc = cross_val_score(RandomForestClassifier(300, random_state=SEED), X, y,
                         cv=cv, groups=groups, n_jobs=-1)
    return ci95(sc)


def sweep_memory():
    base = 60
    out = []
    for r in REL:
        MEM.T = max(6, int(round(base * r)))
        sigs, y = MEM.build_signals(n_per_action=60, seed=SEED)
        m, h = cv_acc(sigs, y, None)
        out.append((r, m, h, len(y)))
        print(f"  memory  T={MEM.T:3d} (x{r}) n={len(y):4d} acc={m:.3f}+/-{h:.3f}", flush=True)
    return out


def sweep_ecology():
    samples = ECO.load_samples()
    base = 20
    out = []
    for r in REL:
        ECO.WIN = max(4, int(round(base * r)))
        sigs, y, blocks = ECO.build_signals_blocked(samples)
        m, h = cv_acc(sigs, y, blocks)
        out.append((r, m, h, len(y)))
        print(f"  ecology win={ECO.WIN:3d} (x{r}) n={len(y):4d} acc={m:.3f}+/-{h:.3f}", flush=True)
    return out


def sweep_graphs():
    base = 1000
    out = []
    for r in REL:
        GR.WIN = int(round(base * r))
        sigs, y, blocks = GR.build_signals_blocked()
        m, h = cv_acc(sigs, y, blocks)
        out.append((r, m, h, len(y)))
        print(f"  graphs  win={GR.WIN:4d} (x{r}) n={len(y):4d} acc={m:.3f}+/-{h:.3f}", flush=True)
    return out


def main():
    print("memory sweep...", flush=True); mem = sweep_memory()
    print("graphs sweep...", flush=True); gr = sweep_graphs()
    print("ecology sweep...", flush=True); eco = sweep_ecology()
    fig, ax = plt.subplots(figsize=(5.6, 4))
    series = [(mem, "tab:blue", "memory (action)"),
              (gr, "tab:green", "dynamic graphs (network)"),
              (eco, "tab:orange", "ecology (region)")]
    for data, color, lab in series:
        r = [d[0] for d in data]; m = np.array([d[1] for d in data]); h = np.array([d[2] for d in data])
        ax.fill_between(r, m - h, m + h, alpha=0.18, color=color)
        ax.plot(r, m, "o-", color=color, label=lab)
    ax.axhline(1 / 3, ls="--", color="0.5", lw=1, label="chance (3 classes)")
    ax.set_xlabel("relative observation window length (window / default)")
    ax.set_ylabel("source-classification accuracy")
    ax.set_title("Accuracy vs observation window length (95% CI)")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.35)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "figures", "F7_window_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    json.dump({"relative": REL,
               "memory": [[round(x, 4) for x in d] for d in mem],
               "graphs": [[round(x, 4) for x in d] for d in gr],
               "ecology": [[round(x, 4) for x in d] for d in eco]},
              open(os.path.join(HERE, "results_window_sweep.json"), "w"), indent=2)
    print("saved", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
