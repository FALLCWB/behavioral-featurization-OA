"""Didactic memory case study for Behavioral Featurization.

A small "program" maintains a memory space (a set of occupied addresses) and mutates
it RANDOMLY by nature at every step: diffuse overwrites plus balanced small
allocations and frees, so the occupancy size random-walks. On top of that natural
background, exactly one ACTION fires as a single event at a random time in each
episode:

    ALLOC  - one allocation burst   (a step up in size, localized)
    FREE   - one free burst          (a step down in size, localized)
    SCAN   - one localized overwrite (size steady, a concentrated change spike)

The memory is dumped at every step. From the behavioral signals of that dump the
recipe identifies which action fired, against the natural random background. This is
a didactic demonstration, not an application; the successful real-world application
is cited (Lemos et al. 2024, JNSM).
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

ADDR_SPACE = 4096
N0 = 2000          # baseline occupancy
T = 60             # time steps per episode
ACTIONS = ["ALLOC", "FREE", "SCAN"]


def natural_step(occupied, rng):
    """Background random activity: diffuse overwrites + balanced small alloc/free
    (size random-walks, net ~ 0). Returns the set of changed addresses."""
    changed = set()
    base = np.array(sorted(occupied)) if occupied else np.array([], dtype=int)
    if base.size:
        w = int(rng.integers(25, 45))
        changed |= set(rng.choice(base, size=min(w, base.size), replace=False).tolist())
    a = int(rng.integers(0, 25))
    new = set(rng.integers(0, ADDR_SPACE, size=a).tolist()) - occupied
    occupied |= new
    f = int(rng.integers(0, 25))
    drop = set()
    if base.size:
        drop = set(rng.choice(base, size=min(f, base.size), replace=False).tolist())
        occupied -= drop
    return changed | new | drop


def action_event(action, occupied, rng):
    """One discrete action event, modest in size so it competes with the noise."""
    changed = set()
    base = np.array(sorted(occupied)) if occupied else np.array([], dtype=int)
    if action == "ALLOC":
        k = int(rng.integers(25, 45))
        start = int(rng.integers(0, ADDR_SPACE - k))
        new = set(range(start, start + k)) - occupied
        occupied |= new
        changed |= new
    elif action == "FREE":
        if base.size > 1:
            k = int(rng.integers(25, 45))
            i = int(rng.integers(0, max(1, base.size - k)))
            drop = set(base[i:i + k].tolist())
            occupied -= drop
            changed |= drop
    elif action == "SCAN":
        if base.size > 1:
            k = int(rng.integers(45, 70))
            i = int(rng.integers(0, max(1, base.size - k)))
            changed |= set(base[i:i + k].tolist())  # localized overwrite, size steady
    return changed


def simulate_episode(action, rng):
    occupied = set(rng.choice(ADDR_SPACE, size=N0, replace=False).tolist())
    t_event = int(rng.integers(20, 40))
    counts, deltas, churns, nets, sp_entropy, centroid = [], [], [], [], [], []
    change_log = []
    for t in range(T):
        before = set(occupied)
        ch = natural_step(occupied, rng)
        if t == t_event:
            ch = ch | action_event(action, occupied, rng)
        added = occupied - before
        removed = before - occupied
        allch = added | removed | ch
        counts.append(len(occupied))
        deltas.append(len(allch))
        churns.append(len(added) + len(removed))
        nets.append(len(added) - len(removed))
        if allch:
            arr = np.fromiter(allch, dtype=float)
            hist, _ = np.histogram(arr, bins=16, range=(0, ADDR_SPACE))
            p = hist[hist > 0] / hist.sum()
            sp_entropy.append(float(-(p * np.log(p)).sum()))
            centroid.append(float(arr.mean()))
            for a in rng.choice(arr, size=min(6, arr.size), replace=False):
                change_log.append((t, float(a)))
        else:
            sp_entropy.append(0.0)
            centroid.append(centroid[-1] if centroid else 0.0)
    signals = {
        "count": counts, "delta": deltas, "churn": churns,
        "net": nets, "spatial_entropy": sp_entropy, "centroid": centroid,
    }
    return signals, change_log, t_event


def build_signals(n_per_action=60, seed=0):
    """Raw per-window behavioral signals (for the contrast / ablation harness)."""
    rng = np.random.default_rng(seed)
    sigs, y = [], []
    for action in ACTIONS:
        for _ in range(n_per_action):
            signals, _, _ = simulate_episode(action, rng)
            sigs.append(signals)
            y.append(action)
    return sigs, np.array(y)


def build_dataset(n_per_action=60, seed=0):
    rng = np.random.default_rng(seed)
    X, y, reps = [], [], {}
    for action in ACTIONS:
        for _ in range(n_per_action):
            signals, _, _ = simulate_episode(action, rng)
            X.append(featurize_signals(signals))
            y.append(action)
        reps[action] = simulate_episode(action, rng)
    return np.array(X), np.array(y), reps


def make_figure(reps, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"ALLOC": "tab:blue", "FREE": "tab:red", "SCAN": "tab:green"}
    for action, (sig, _, te) in reps.items():
        axes[0].plot(sig["count"], label=action, color=colors[action], lw=1.8)
        axes[0].axvline(te, color=colors[action], ls=":", alpha=0.5)
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("|O_t|  (occupied addresses)")
    axes[0].set_title("Memory dump: size over time (natural churn + action event)")
    axes[0].legend()
    for action, (_, log, _) in reps.items():
        if log:
            ts = [t for t, _ in log]
            ad = [a for _, a in log]
            axes[1].scatter(ts, ad, s=8, color=colors[action], label=action, alpha=0.5)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("changed address")
    axes[1].set_title("Where change occurs (diffuse natural + action signature)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    here = os.path.dirname(__file__)
    figdir = os.path.join(here, "..", "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    X, y, reps = build_dataset()
    res = run_classification(X, y)
    print("=== Behavioral Featurization - memory didactic demo (natural churn + 1 action event) ===")
    print(f"episodes: {len(y)} | classes: {res['labels']} | features per object: {X.shape[1]}")
    print(f"CV accuracy: {res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")
    print(f"confusion (rows=true {res['labels']}, cols=pred): {res['confusion']}")
    print(res["report"])
    figpath = os.path.join(figdir, "F1_memory_demo.png")
    make_figure(reps, figpath)
    print("figure saved:", os.path.relpath(figpath, here))
    with open(os.path.join(here, "results_memory.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)


if __name__ == "__main__":
    main()
