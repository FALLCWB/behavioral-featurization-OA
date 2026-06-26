"""Contrast with the giants + ablation for Behavioral Featurization.

Contrast: per domain, compare the full Behavioral Featurization representation
against (a) a Deep Sets-style permutation-invariant pool, (b) a synopsis/sketch
quantile summary, and (c) a naive final-snapshot baseline. Set-pool and sketch are
order-invariant (they see the multiset of values, not the temporal arrangement),
which is exactly the marginal-only regime the Proposition predicts will lose signal
where temporal structure matters.

Ablation: per domain, drop the dispersion family ("where the change concentrates"
analog) and measure the accuracy change, plus the dispersion-only accuracy; then a
Benjamini-Hochberg FDR feature selection at alpha=0.05.

Same RandomForest, repeated stratified CV across all conditions.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFdr, f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import real_memory_demo as memory_demo
import graphs_demo
import ecology_demo
from pipeline import featurize_signals, pool_features, sketch_features, naive_features

REPEATS = 10
SEED = 0


def rcv(X, y, trees=200):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED)
    s = cross_val_score(RandomForestClassifier(n_estimators=trees, random_state=SEED),
                        X, y, cv=rkf, n_jobs=-1)
    return float(s.mean()), float(s.std())


def subset(sig, keep=None, drop=None):
    if keep is not None:
        return {k: v for k, v in sig.items() if k in keep}
    if drop is not None:
        return {k: v for k, v in sig.items() if k not in drop}
    return sig


def run_domain(name, sigs, y, dispersion):
    y = np.asarray(y)
    chance = 1.0 / len(set(y.tolist()))
    Xb = np.array([featurize_signals(s) for s in sigs])
    out = {"domain": name, "chance": chance, "contrast": {}, "ablation": {}, "fdr": {}}

    print(f"\n### {name}   (chance={chance:.3f})")
    print("-- Contrast with the giants --")
    reps = {
        "Behavioral Featurization (ours)": Xb,
        "Set-pool (Deep Sets-style)": np.array([pool_features(s) for s in sigs]),
        "Sketch (synopsis)": np.array([sketch_features(s) for s in sigs]),
        "Naive (final snapshot)": np.array([naive_features(s) for s in sigs]),
    }
    for rn, X in reps.items():
        m, sd = rcv(X, y)
        out["contrast"][rn] = [m, sd]
        print(f"  {rn:34s}: {m:.3f} +/- {sd:.3f}")

    print(f"-- Ablation (dispersion family = {sorted(dispersion)}) --")
    abl = {
        "full": Xb,
        "full minus dispersion": np.array([featurize_signals(subset(s, drop=dispersion)) for s in sigs]),
        "dispersion only": np.array([featurize_signals(subset(s, keep=dispersion)) for s in sigs]),
    }
    for rn, X in abl.items():
        m, sd = rcv(X, y)
        out["ablation"][rn] = [m, sd]
        print(f"  {rn:24s}: {m:.3f} +/- {sd:.3f}")
    out["ablation"]["dispersion_marginal_gain"] = round(
        out["ablation"]["full"][0] - out["ablation"]["full minus dispersion"][0], 4)
    print(f"  -> dispersion marginal gain: {out['ablation']['dispersion_marginal_gain']:+.3f}")

    sel = SelectFdr(f_classif, alpha=0.05).fit(Xb, y)
    nsel = int(sel.get_support().sum())
    m, sd = rcv(sel.transform(Xb), y)
    out["fdr"] = {"n_selected": nsel, "n_total": int(Xb.shape[1]), "acc": [m, sd]}
    print(f"-- FDR selection (alpha=0.05): kept {nsel}/{Xb.shape[1]} features, acc {m:.3f} +/- {sd:.3f}")
    return out


def main():
    results = []

    sm, ym = memory_demo.build_signals()
    results.append(run_domain("Memory (real)", sm, ym, {"spatial_entropy", "centroid"}))

    sg, yg = graphs_demo.build_signals()
    results.append(run_domain("Dynamic graphs", sg, yg, {"density", "new_node_rate"}))

    print("\nloading ecology samples...")
    samples = ecology_demo.load_samples()
    se, ye = ecology_demo.build_signals(samples)
    results.append(run_domain("Ecology (EcoMon)", se, ye, {"turnover", "new_taxa_rate"}))

    with open(os.path.join(HERE, "results_contrast_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved: results_contrast_ablation.json")


if __name__ == "__main__":
    main()
