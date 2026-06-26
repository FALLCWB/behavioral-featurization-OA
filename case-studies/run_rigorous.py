"""Rigorous, leakage-free re-evaluation addressing the peer review.

For each domain:
  - leakage-free CV: memory uses StratifiedKFold (episodes are independent draws);
    graphs and ecology use StratifiedGroupKFold over CONTIGUOUS TIME BLOCKS with
    NON-overlapping windows, so no adjacent/overlapping window crosses a train/test
    split (fixes the train/test leakage flagged in review).
  - representations: behavioral (ours), set-pool, sketch, naive snapshot, and the
    order-aware PATH SIGNATURE baseline.
  - ordered-vs-shuffled DIAGNOSTIC: does temporal order carry real signal?
  - triviality check: can a single behavioral feature match the full accuracy?
  - permutation test (1000 permutations) under the same CV scheme.
Writes results_rigorous.json.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import real_memory_demo as memory_demo
import graphs_demo
import ecology_demo
from pipeline import (featurize_signals, pool_features, sketch_features,
                      naive_features, signature_features, shuffle_signals)

PERMS = 1000
SEED = 0


def make_cv(blocks):
    if blocks is None:
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), None
    return StratifiedGroupKFold(n_splits=5), blocks


def cv_acc(X, y, cv, groups, trees=300):
    m = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    s = cross_val_score(m, X, y, cv=cv, groups=groups, n_jobs=-1)
    return float(s.mean()), float(s.std())


def perm_test(X, y, cv, groups, obs, perms=PERMS, trees=150):
    rng = np.random.default_rng(SEED)
    m = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    null = np.empty(perms)
    for i in range(perms):
        yp = rng.permutation(y)
        null[i] = cross_val_score(m, X, yp, cv=cv, groups=groups, n_jobs=-1).mean()
    p = (np.sum(null >= obs) + 1) / (perms + 1)
    return float(p), float(null.max())


def feat(sigs, fn):
    return np.array([fn(s) for s in sigs])


def run_domain(name, sigs, y, blocks):
    y = np.asarray(y)
    cv, groups = make_cv(blocks)
    rng = np.random.default_rng(SEED)
    classes = sorted(set(y.tolist()))
    chance = 1.0 / len(classes)

    Xb = feat(sigs, featurize_signals)
    reps = {
        "behavioral": Xb,
        "set_pool": feat(sigs, pool_features),
        "sketch": feat(sigs, sketch_features),
        "naive_snapshot": feat(sigs, naive_features),
        "signature_order_aware": feat(sigs, signature_features),
        "behavioral_shuffled": np.array([featurize_signals(shuffle_signals(s, rng)) for s in sigs]),
    }

    out = {"domain": name, "n": int(len(y)), "classes": classes, "chance": chance,
           "cv": ("StratifiedKFold" if blocks is None else "StratifiedGroupKFold(time-blocks)"),
           "rep_acc": {}}
    print(f"\n### {name}  | n={len(y)} | classes={classes} | chance={chance:.3f} | CV={out['cv']}")
    for rn, X in reps.items():
        m, sd = cv_acc(X, y, cv, groups)
        out["rep_acc"][rn] = [m, sd]
        print(f"  {rn:24s}: {m:.3f} +/- {sd:.3f}")

    gap = out["rep_acc"]["behavioral"][0] - out["rep_acc"]["behavioral_shuffled"][0]
    out["order_gap"] = round(gap, 4)
    print(f"  -> ordered-vs-shuffled gap (does order matter?): {gap:+.3f}")

    # triviality: top-3 single behavioral features by ANOVA F, CV each with a shallow tree
    F, _ = f_classif(np.nan_to_num(Xb), y)
    top = np.argsort(np.nan_to_num(F))[::-1][:3]
    best1 = 0.0
    for j in top:
        m = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED),
                            Xb[:, [j]], y, cv=cv, groups=groups, n_jobs=-1).mean()
        best1 = max(best1, float(m))
    out["best_single_feature_acc"] = round(best1, 4)
    print(f"  -> best single-feature accuracy (triviality check): {best1:.3f}")

    obs = out["rep_acc"]["behavioral"][0]
    p, nullmax = perm_test(Xb, y, cv, groups, obs)
    out["perm_p"] = p
    out["perm_null_max"] = nullmax
    print(f"  -> permutation test ({PERMS} perms): null max={nullmax:.3f}, p={p:.4f}")
    return out


def main():
    results = []
    sm, ym = memory_demo.build_signals()
    results.append(run_domain("Memory (real)", sm, ym, None))

    sg, yg, bg = graphs_demo.build_signals_blocked()
    results.append(run_domain("Dynamic graphs", sg, yg, bg))

    print("\nloading ecology samples...")
    samples = ecology_demo.load_samples()
    se, ye, be = ecology_demo.build_signals_blocked(samples)
    results.append(run_domain("Ecology (EcoMon)", se, ye, be))

    with open(os.path.join(HERE, "results_rigorous.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved: results_rigorous.json")


if __name__ == "__main__":
    main()
