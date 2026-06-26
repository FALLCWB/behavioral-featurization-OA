"""Motivation figures for Behavioral Featurization.

F0a: a memory dump over time (address x time occupancy heatmap + size curve),
     showing the object literally changing size and shape.
F0b: the difficulty we claim, across all three domains: the raw dimensionality
     d_t (occupied addresses / active graph nodes / taxon richness) is not fixed,
     so no fixed-length input vector exists for naive ML. Graphs and ecology are
     real data.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import real_memory_demo as mem
import graphs_demo as gr
import ecology_demo as eco

FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def memory_dump_episode(action="ALLOC", bins=64, seed=7):
    """Re-run one REAL episode capturing the full per-step occupancy. Each step the
    arena is dumped from /proc/self/mem and binned by address; occupancy is the
    count of nonzero (live) bytes read back from real RAM."""
    rng = np.random.default_rng(seed)
    arena = mem.RealArena(rng)
    mem._fill_baseline(arena)
    t_event = int(rng.integers(max(2, int(0.34 * mem.T)), max(3, int(0.66 * mem.T))))
    dump = np.zeros((mem.T, bins))
    dt = []
    for t in range(mem.T):
        mem.natural_step(arena)
        if t == t_event:
            mem.action_event(action, arena)
        raw = arena.dump()                      # real bytes via /proc/self/mem
        nz = (raw != 0).astype(float)
        dump[t] = nz.reshape(bins, -1).sum(axis=1)   # occupied bytes per address bin
        dt.append(float(nz.sum()) / 1024.0)          # occupancy in KiB
    arena.close()
    return dump, np.array(dt), t_event


def fig_memory_dump():
    dump, dt, te = memory_dump_episode(action="ALLOC")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(dump.T, aspect="auto", origin="lower",
                        extent=[0, mem.T, 0, mem.ARENA / 1024], cmap="viridis")
    axes[0].axvline(te, color="white", ls=":", lw=1.5)
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("address (KiB)")
    axes[0].set_title("occupancy by address")
    fig.colorbar(im, ax=axes[0], label="live bytes / address bin")
    axes[1].plot(dt, color="tab:blue", lw=2)
    axes[1].axvline(te, color="gray", ls=":", lw=1.5)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("occupancy (KiB)")
    axes[1].set_title("total occupancy")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "F0a_memory_dump.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_volatile_dimensionality():
    # memory: occupied count over one episode
    _, dt_mem, _ = memory_dump_episode(action="SCAN", seed=3)
    # graphs (real): active node count over one window
    edges = gr.load_edges(os.path.join(HERE, "graphs-dynamic", "data", "CollegeMsg.txt"))
    g_nodes = gr.window_signals(edges[5000:5000 + gr.WIN])["nodes"]
    # ecology (real): taxon richness over one window
    samples = eco.load_samples()
    north = sorted((s for s in samples if eco.region_of(s["lat"]) == "north"),
                   key=lambda s: s["date"])
    e_rich = eco.window_signals(north[200:200 + eco.WIN])["richness"]

    panels = [
        ("Memory (occupancy, KiB) - real", dt_mem, "tab:blue"),
        ("Dynamic graph (active nodes) - real", g_nodes, "tab:green"),
        ("Ecology (taxon richness) - real", e_rich, "tab:orange"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (title, series, color) in zip(axes, panels):
        s = np.asarray(series, dtype=float)
        ax.plot(s, color=color, lw=2)
        ax.fill_between(range(len(s)), s.min(), s, color=color, alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time step")
        ax.set_ylabel("dimensionality (component count)")
        ax.text(0.5, 0.92, f"range {int(s.min())}–{int(s.max())}",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="black",
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.8))
    fig.tight_layout()
    out = os.path.join(FIGDIR, "F0b_volatile_dimensionality.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    a = fig_memory_dump()
    print("saved:", os.path.relpath(a, HERE))
    b = fig_volatile_dimensionality()
    print("saved:", os.path.relpath(b, HERE))


if __name__ == "__main__":
    main()
