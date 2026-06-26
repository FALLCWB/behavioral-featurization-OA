"""Per-domain feature selection, done correctly (selection INSIDE the CV fold).

Compares the compact 16-feature behavioral set against a rich candidate pool
(tsfresh EfficientFCParameters on the same behavioral signals) followed by
principled feature selection performed within each training fold (no selection
leakage). Question (Filipe): does the ecology ~0.63 improve with selected features?

Leakage controls preserved: non-overlapping windows + blocked group CV for
graphs/ecology, stratified k-fold for the independent memory episodes.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import real_memory_demo as memory_demo, graphs_demo, ecology_demo
from pipeline import featurize_signals, tsfresh_featurize_batch
from run_rigorous import make_cv, cv_acc

KS = [20, 50, 100, 200]


def eval_select(X, y, cv, groups, k):
    pipe = Pipeline([
        ("sel", SelectKBest(f_classif, k=min(k, X.shape[1]))),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=0)),
    ])
    s = cross_val_score(pipe, X, y, cv=cv, groups=groups, n_jobs=-1)
    return float(s.mean()), float(s.std())


def run(name, sigs, y, blocks):
    y = np.asarray(y)
    cv, groups = make_cv(blocks)
    out = {"domain": name}
    Xc = np.array([featurize_signals(s) for s in sigs])
    m, sd = cv_acc(Xc, y, cv, groups)
    out["compact_16"] = [round(m, 4), round(sd, 4)]
    print(f"\n### {name} | n={len(y)}")
    print(f"  compact 16-feature behavioral (baseline): {m:.3f} +/- {sd:.3f}", flush=True)
    print("  rich pool = tsfresh(efficient) on the behavioral signals, selection INSIDE CV:", flush=True)
    Xr = np.nan_to_num(tsfresh_featurize_batch(sigs)[0])
    out["rich_pool_size"] = int(Xr.shape[1])
    out["rich_select"] = {}
    for k in KS:
        mk, sdk = eval_select(Xr, y, cv, groups, k)
        out["rich_select"][str(k)] = [round(mk, 4), round(sdk, 4)]
        print(f"    SelectKBest k={k:4d} (of {Xr.shape[1]}): {mk:.3f} +/- {sdk:.3f}", flush=True)
    best_k = max(out["rich_select"], key=lambda kk: out["rich_select"][kk][0])
    out["best"] = {"k": best_k, "acc": out["rich_select"][best_k][0],
                   "delta_vs_compact": round(out["rich_select"][best_k][0] - out["compact_16"][0], 4)}
    print(f"  -> best selected: k={best_k}, acc={out['best']['acc']:.3f}, "
          f"delta vs compact = {out['best']['delta_vs_compact']:+.3f}", flush=True)
    return out


def main():
    res = []
    sm, ym = memory_demo.build_signals()
    res.append(run("Memory (real)", sm, ym, None))
    sg, yg, bg = graphs_demo.build_signals_blocked()
    res.append(run("Dynamic graphs", sg, yg, bg))
    samples = ecology_demo.load_samples()
    se, ye, be = ecology_demo.build_signals_blocked(samples)
    res.append(run("Ecology (EcoMon)", se, ye, be))
    json.dump(res, open(os.path.join(HERE, "results_feature_selection.json"), "w"), indent=2)
    print("\nsaved results_feature_selection.json", flush=True)


if __name__ == "__main__":
    main()
