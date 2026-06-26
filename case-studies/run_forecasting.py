"""Forecasting: a REAL task whose natural label depends on temporal dynamics.

For each source (network / region), windows are ordered in time. The features come
from the CURRENT window; the label is the direction of the next window's primary
signal (does its mean go up or down). This is a natural, non-constructed label (the
real future), and it should depend on the within-window dynamics (trend/momentum),
not only on the marginal. Comparing the order-retaining representations against the
marginal-only ones, and the ordered-vs-shuffled behavioral gap, tests directly
whether retaining temporal order helps a real predictive task.

Leakage controlled: non-overlapping windows + StratifiedGroupKFold over time blocks.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import graphs_demo as gr
import ecology_demo as eco
from pipeline import (featurize_signals, pool_features, sketch_features,
                      naive_features, signature_features, shuffle_signals)

SEED = 0
N_BLOCKS = 5


def graph_sources():
    out = {}
    for name, fn in gr.NETS.items():
        edges = gr.load_edges(os.path.join(HERE, "graphs-dynamic", "data", fn))
        nwin = len(edges) // gr.WIN
        out[name] = [gr.window_signals(edges[i * gr.WIN:(i + 1) * gr.WIN]) for i in range(nwin)]
    return out


def ecology_sources():
    samples = eco.load_samples()
    by = {"south": [], "mid": [], "north": []}
    for s in samples:
        by[eco.region_of(s["lat"])].append(s)
    out = {}
    for region, lst in by.items():
        lst.sort(key=lambda s: s["date"])
        nwin = len(lst) // eco.WIN
        out[region] = [eco.window_signals(lst[i * eco.WIN:(i + 1) * eco.WIN]) for i in range(nwin)]
    return out


def make_forecast(sources, signal_key):
    """Features = current window; label = sign of next window's mean change."""
    sigs, y, block = [], [], []
    for src, wins in sources.items():
        means = [float(np.mean(w[signal_key])) for w in wins]
        m = len(wins) - 1
        for i in range(m):
            sigs.append(wins[i])
            y.append(1 if means[i + 1] > means[i] else 0)
            block.append(min(N_BLOCKS - 1, i * N_BLOCKS // max(m, 1)))
    return sigs, np.array(y), np.array(block)


def cv(X, y, groups, trees=400):
    m = RandomForestClassifier(n_estimators=trees, random_state=SEED)
    s = cross_val_score(m, X, y, cv=StratifiedGroupKFold(5), groups=groups, n_jobs=-1)
    return float(s.mean()), float(s.std())


def run(name, sources, signal_key):
    rng = np.random.default_rng(SEED)
    sigs, y, block = make_forecast(sources, signal_key)
    base = max(np.mean(y == 0), np.mean(y == 1))  # majority-class baseline
    reps = {
        "behavioral": np.array([featurize_signals(s) for s in sigs]),
        "signature_order": np.array([signature_features(s) for s in sigs]),
        "set_pool": np.array([pool_features(s) for s in sigs]),
        "sketch": np.array([sketch_features(s) for s in sigs]),
        "naive_snapshot": np.array([naive_features(s) for s in sigs]),
        "behavioral_shuffled": np.array([featurize_signals(shuffle_signals(s, rng)) for s in sigs]),
    }
    out = {"domain": name, "signal": signal_key, "n": int(len(y)),
           "majority_baseline": round(base, 4), "acc": {}}
    print(f"\n### Forecast next-window direction: {name} ({signal_key}) | n={len(y)} | majority={base:.3f}")
    for rn, X in reps.items():
        a, sd = cv(X, y, block)
        out["acc"][rn] = [round(a, 4), round(sd, 4)]
        print(f"  {rn:22s}: {a:.3f} +/- {sd:.3f}")
    out["order_gap"] = round(out["acc"]["behavioral"][0] - out["acc"]["behavioral_shuffled"][0], 4)
    out["behavioral_minus_pool"] = round(out["acc"]["behavioral"][0] - out["acc"]["set_pool"][0], 4)
    print(f"  -> order gap (behavioral - shuffled): {out['order_gap']:+.3f}")
    print(f"  -> behavioral - set_pool: {out['behavioral_minus_pool']:+.3f}")
    return out


def main():
    res = []
    res.append(run("Dynamic graphs", graph_sources(), "nodes"))
    res.append(run("Ecology (EcoMon)", ecology_sources(), "richness"))
    json.dump(res, open(os.path.join(HERE, "results_forecasting.json"), "w"), indent=2)
    print("\nsaved results_forecasting.json")


if __name__ == "__main__":
    main()
