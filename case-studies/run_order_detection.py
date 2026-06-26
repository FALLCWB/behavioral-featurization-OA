"""Order-structure detection on REAL data.

Directly tests whether the temporal axis carries real, detectable structure in each
domain's evolving object, independent of the source-classification label. For each
window, the task is to distinguish the real (ordered) signal from a time-perturbed
copy of itself:
  - shuffle: time index permuted (destroys all temporal order; multiset preserved)
  - reverse: time reversed (flips direction; magnitude-symmetric statistics preserved)

By construction, set-pool and sketch (marginal-only) are at chance for shuffle (the
multiset is identical), so any above-chance accuracy from the behavioral or signature
representation is proof that the order axis carries real structure the marginal-only
methods discard. This is the real-data counterpart of the synthetic generativity test.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import real_memory_demo as memory_demo
import graphs_demo
import ecology_demo
from pipeline import (featurize_signals, pool_features, sketch_features,
                      signature_features, shuffle_signals)

SEED = 0


def reverse_signals(s):
    return {k: list(np.asarray(v, dtype=float)[::-1]) for k, v in s.items()}


def build_detection(sigs, blocks, mode, rng):
    """PAIRED design: each window contributes BOTH its ordered version and its own
    time-perturbed version, with the same group/block. Because the perturbation
    preserves the window's multiset exactly, the marginal-only representations
    (set-pool, sketch) produce IDENTICAL features for the two versions of a window,
    so they are at chance by construction; only order-dependent features can tell
    them apart."""
    items, labels, grp = [], [], []
    for i, s in enumerate(sigs):
        # group = pair id: a window and its perturbed twin share a group, so they
        # never split across folds (otherwise an identical marginal-only twin in the
        # training fold flips predictions and pushes set-pool/sketch below chance).
        items.append(s); labels.append("ordered"); grp.append(i)
        pert = shuffle_signals(s, rng) if mode == "shuffle" else reverse_signals(s)
        items.append(pert); labels.append("perturbed"); grp.append(i)
    return items, np.array(labels), np.array(grp)


def cv_acc(X, y, groups, trees=300):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED) if groups is None \
        else StratifiedGroupKFold(5)
    m = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    s = cross_val_score(m, X, y, cv=cv, groups=groups, n_jobs=-1)
    return float(s.mean()), float(s.std())


def run_domain(name, sigs, blocks):
    rng = np.random.default_rng(SEED)
    out = {"domain": name, "n_windows": len(sigs), "chance": 0.5, "modes": {}}
    print(f"\n### {name} | windows={len(sigs)} | chance=0.500")
    reps = {
        "behavioral": featurize_signals,
        "signature_order_aware": signature_features,
        "set_pool": pool_features,
        "sketch": sketch_features,
    }
    for mode in ["shuffle", "reverse"]:
        items, y, grp = build_detection(sigs, blocks, mode, rng)
        out["modes"][mode] = {}
        print(f"  -- detect {mode} vs ordered --")
        for rn, fn in reps.items():
            X = np.array([fn(s) for s in items])
            m, sd = cv_acc(X, y, grp)
            out["modes"][mode][rn] = [m, sd]
            print(f"     {rn:22s}: {m:.3f} +/- {sd:.3f}")
    return out


def main():
    results = []
    sm, _ = memory_demo.build_signals()
    results.append(run_domain("Memory (real)", sm, None))
    sg, _, bg = graphs_demo.build_signals_blocked()
    results.append(run_domain("Dynamic graphs", sg, bg))
    print("\nloading ecology samples...")
    samples = ecology_demo.load_samples()
    se, _, be = ecology_demo.build_signals_blocked(samples)
    results.append(run_domain("Ecology (EcoMon)", se, be))
    with open(os.path.join(HERE, "results_order_detection.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved: results_order_detection.json")


if __name__ == "__main__":
    main()
