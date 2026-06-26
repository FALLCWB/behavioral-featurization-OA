"""Computational cost of the technique and of each representation.

For each domain and each representation (behavioral / set-pool / sketch / signature /
tsfresh), measures: the fixed-vector DIMENSION, the FEATURIZATION time (total and per
window), and the downstream model FIT time on the resulting vectors. This quantifies
the trade the practitioner faces: a small fixed vector is cheap to build and to learn
on, while a comprehensive feature library is far larger and slower for the same task.
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import real_memory_demo as memory_demo, graphs_demo, ecology_demo
from pipeline import (featurize_signals, pool_features, sketch_features,
                      signature_features, tsfresh_featurize_batch)

REPS = {
    "behavioral": featurize_signals,
    "set_pool": pool_features,
    "sketch": sketch_features,
    "signature": signature_features,
}


def best_time(fn, reps=3):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def run(name, sigs, y):
    y = np.asarray(y)
    n = len(sigs)
    out = {"domain": name, "n_windows": n, "reps": {}}
    print(f"\n### {name} | {n} windows", flush=True)
    for rn, fn in REPS.items():
        dim = len(fn(sigs[0]))
        ft = best_time(lambda: [fn(s) for s in sigs])
        X = np.array([fn(s) for s in sigs])
        clf = RandomForestClassifier(n_estimators=300, random_state=0)
        mt = best_time(lambda: clf.fit(X, y), reps=1)
        out["reps"][rn] = {"dim": int(dim), "featurize_ms_total": round(ft * 1e3, 1),
                           "featurize_us_per_window": round(ft / n * 1e6, 1),
                           "fit_s": round(mt, 2)}
        print(f"  {rn:11s}: dim={dim:5d}  featurize={ft*1e3:7.1f} ms "
              f"({ft/n*1e6:6.1f} us/win)  fit={mt:5.2f} s", flush=True)
    # tsfresh (rich library) measured separately (slow)
    t0 = time.perf_counter()
    Xr, _ = tsfresh_featurize_batch(sigs)
    ft = time.perf_counter() - t0
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    t1 = time.perf_counter()
    clf.fit(np.nan_to_num(Xr), y)
    mt = time.perf_counter() - t1
    out["reps"]["tsfresh"] = {"dim": int(Xr.shape[1]), "featurize_ms_total": round(ft * 1e3, 1),
                              "featurize_us_per_window": round(ft / n * 1e6, 1), "fit_s": round(mt, 2)}
    print(f"  {'tsfresh':11s}: dim={Xr.shape[1]:5d}  featurize={ft*1e3:7.1f} ms "
          f"({ft/n*1e6:6.1f} us/win)  fit={mt:5.2f} s", flush=True)
    return out


def main():
    res = []
    sm, ym = memory_demo.build_signals()
    res.append(run("Memory", sm, ym))
    sg, yg, _ = graphs_demo.build_signals_blocked()
    res.append(run("Dynamic graphs", sg, yg))
    samples = ecology_demo.load_samples()
    se, ye, _ = ecology_demo.build_signals_blocked(samples)
    res.append(run("Ecology", se, ye))
    json.dump(res, open(os.path.join(HERE, "results_cost.json"), "w"), indent=2)
    print("\nsaved results_cost.json", flush=True)


if __name__ == "__main__":
    main()
