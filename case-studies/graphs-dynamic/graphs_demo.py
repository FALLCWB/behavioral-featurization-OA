"""Dynamic-graph case study for Behavioral Featurization.

Three real temporal networks (SNAP: CollegeMsg, email-Eu-core-temporal,
sx-mathoverflow) are streamed in time order and sliced into windows. Inside each
window the ACTIVE graph evolves: edges arrive and expire over a trailing horizon, so
the active node and edge sets are both unknown a priori and volatile over time
(the both-at-once condition). The same shared recipe derives behavioral signals from
that evolution and classifies which network a window came from.

Identical pipeline to the memory case study (case-studies/common/pipeline.py):
reusing the same code across unrelated domains is the generality argument.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pipeline import featurize_signals, run_classification

NETS = {
    "college": "CollegeMsg.txt",
    "email": "email-Eu-core-temporal.txt",
    "mathoverflow": "sx-mathoverflow.txt",
}
MAX_EDGES = 45000   # cap per network for a light, fast demo
WIN = 1000          # edges per window
STEP = 500          # slide
K = 30              # sub-bins (time steps) per window
HORIZON = 6         # sub-bins an edge stays "active" before it expires


def load_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                edges.append((int(p[0]), int(p[1]), int(p[2])))
    edges.sort(key=lambda r: r[2])
    return edges[:MAX_EDGES]


def window_signals(edges):
    """Evolve the active graph across K sub-bins; emit behavioral signals over time."""
    n = len(edges)
    per = max(1, n // K)
    bins = [edges[i * per:(i + 1) * per] for i in range(K)]
    bin_edge_sets = []
    seen_nodes = set()
    sig = {k: [] for k in ["nodes", "edges", "churn", "density", "mean_deg", "new_node_rate"]}
    for b in bins:
        es = set((min(u, v), max(u, v)) for u, v, _ in b)
        bin_edge_sets.append(es)
        recent = bin_edge_sets[-HORIZON:]
        active_edges = set().union(*recent) if recent else set()
        active_nodes = set()
        for (u, v) in active_edges:
            active_nodes.add(u)
            active_nodes.add(v)
        ne, nn = len(active_edges), len(active_nodes)
        nodes_this = set()
        for u, v, _ in b:
            nodes_this.add(u)
            nodes_this.add(v)
        new_nodes = nodes_this - seen_nodes
        seen_nodes |= nodes_this
        sig["nodes"].append(nn)
        sig["edges"].append(ne)
        sig["churn"].append(len(es))
        sig["density"].append(ne / (nn * (nn - 1) / 2 + 1e-9) if nn > 1 else 0.0)
        sig["mean_deg"].append(2 * ne / nn if nn > 0 else 0.0)
        sig["new_node_rate"].append(len(new_nodes) / (len(nodes_this) + 1e-9))
    return sig


def build_signals_blocked(n_blocks=5):
    """Leakage-free version: NON-overlapping windows (stride = window), and a
    time-block id per window so cross-validation can hold out whole contiguous time
    blocks of each network (no adjacent/overlapping window spans a train/test split)."""
    here = os.path.dirname(__file__)
    sigs, y, block = [], [], []
    for name, fn in NETS.items():
        edges = load_edges(os.path.join(here, "data", fn))
        nwin = len(edges) // WIN
        for i in range(nwin):
            sigs.append(window_signals(edges[i * WIN:(i + 1) * WIN]))
            y.append(name)
            block.append(min(n_blocks - 1, i * n_blocks // nwin))
    return sigs, np.array(y), np.array(block)


def build_signals():
    """Raw per-window behavioral signals (for the contrast / ablation harness)."""
    here = os.path.dirname(__file__)
    sigs, y = [], []
    for name, fn in NETS.items():
        edges = load_edges(os.path.join(here, "data", fn))
        i = 0
        while i + WIN <= len(edges):
            sigs.append(window_signals(edges[i:i + WIN]))
            y.append(name)
            i += STEP
    return sigs, np.array(y)


def build_dataset():
    here = os.path.dirname(__file__)
    X, y, reps = [], [], {}
    counts = {}
    for name, fn in NETS.items():
        edges = load_edges(os.path.join(here, "data", fn))
        i, first, c = 0, True, 0
        while i + WIN <= len(edges):
            sig = window_signals(edges[i:i + WIN])
            X.append(featurize_signals(sig))
            y.append(name)
            if first:
                reps[name] = sig
                first = False
            i += STEP
            c += 1
        counts[name] = c
    return np.array(X), np.array(y), reps, counts


def make_figure(reps, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"college": "tab:blue", "email": "tab:red", "mathoverflow": "tab:green"}
    for name, sig in reps.items():
        axes[0].plot(sig["nodes"], label=name, color=colors[name], lw=1.8)
        axes[1].plot(sig["mean_deg"], label=name, color=colors[name], lw=1.8)
    axes[0].set_xlabel("sub-step within window")
    axes[0].set_ylabel("active node count")
    axes[0].set_title("Dynamic graph: active nodes over time, by network")
    axes[0].legend()
    axes[1].set_xlabel("sub-step within window")
    axes[1].set_ylabel("mean degree")
    axes[1].set_title("Active mean degree over time, by network")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    here = os.path.dirname(__file__)
    figdir = os.path.join(here, "..", "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    X, y, reps, counts = build_dataset()
    res = run_classification(X, y)
    print("=== Behavioral Featurization - dynamic graphs (3 real SNAP temporal networks) ===")
    print(f"windows per network: {counts}")
    print(f"total windows: {len(y)} | classes: {res['labels']} | features per object: {X.shape[1]}")
    print(f"CV accuracy: {res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")
    print(f"confusion (rows=true {res['labels']}, cols=pred): {res['confusion']}")
    print(res["report"])
    figpath = os.path.join(figdir, "F4a_graphs_demo.png")
    make_figure(reps, figpath)
    print("figure saved:", os.path.relpath(figpath, here))
    with open(os.path.join(here, "results_graphs.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)


if __name__ == "__main__":
    main()
