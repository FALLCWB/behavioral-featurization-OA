"""Statistical rigor for the Behavioral Featurization case studies.

For each domain: repeated stratified cross-validation (95% confidence interval on
accuracy) and a label-permutation test (p-value against shuffled labels) to
establish that accuracy is significantly above chance. Same shared recipe / model
across all domains.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import real_memory_demo as memory_demo
import graphs_demo
import ecology_demo
import generativity_synthetic as gen

REPEATS = 20
PERMS = 200
SEED = 0


def repeated_cv(X, y, trees=300):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED)
    model = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    return cross_val_score(model, X, y, cv=rkf, n_jobs=-1)


def perm_test(X, y, trees=150):
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    obs = cross_val_score(model, X, y, cv=skf, n_jobs=-1).mean()
    null = np.array([cross_val_score(model, X, rng.permutation(y), cv=skf, n_jobs=-1).mean()
                     for _ in range(PERMS)])
    p = (np.sum(null >= obs) + 1) / (PERMS + 1)
    return obs, null, p


def evaluate(name, X, y):
    y = np.asarray(y)
    n_classes = len(set(y.tolist()))
    chance = 1.0 / n_classes
    scores = repeated_cv(X, y)
    mean = scores.mean()
    lo, hi = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    obs, null, p = perm_test(X, y)
    sig95 = p < 0.05
    sig98 = p < 0.02
    print(f"\n### {name}")
    print(f"  n={len(y)}  classes={n_classes}  chance={chance:.3f}")
    print(f"  accuracy (repeated 5x{REPEATS} CV): mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
    print(f"  permutation test ({PERMS} perms): null max={null.max():.3f}  p={p:.4f}")
    print(f"  significant >95% conf: {sig95}   >98% conf: {sig98}")
    return {
        "name": name, "n": int(len(y)), "classes": n_classes, "chance": chance,
        "acc_mean": float(mean), "ci95_low": float(lo), "ci95_high": float(hi),
        "perm_p": float(p), "perm_null_max": float(null.max()),
        "sig_95": bool(sig95), "sig_98": bool(sig98),
    }


def main():
    results = []

    Xm, ym, _ = memory_demo.build_dataset()
    results.append(evaluate("Memory (real): identify action", Xm, ym))

    Xg, yg, _, _ = graphs_demo.build_dataset()
    results.append(evaluate("Dynamic graphs (SNAP): identify network", Xg, yg))

    print("\nloading ecology samples...")
    samples = ecology_demo.load_samples()
    Xe, ye, _, _ = ecology_demo.build_dataset(samples)
    results.append(evaluate("Ecology (EcoMon): identify region", Xe, ye))

    seqs, yc = gen.build()
    Xb = np.array([gen.behavioral_feature(s) for s in seqs])
    results.append(evaluate("Generativity: behavioral features (ordered vs shuffled)", Xb, yc))

    with open(os.path.join(HERE, "results_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved:", os.path.relpath(os.path.join(HERE, "results_stats.json"), HERE))


if __name__ == "__main__":
    main()
