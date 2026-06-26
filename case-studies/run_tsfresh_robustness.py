"""Robustness to the featurizer: re-run the source-classification accuracy with an
ESTABLISHED featurizer (tsfresh, EfficientFCParameters) instead of the compact
16-feature set, under the same leakage-controlled CV. Confirms the conclusions do
not depend on the bespoke feature set."""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.feature_selection import SelectFdr, f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import real_memory_demo as memory_demo, graphs_demo, ecology_demo
from pipeline import tsfresh_featurize_batch
from run_rigorous import make_cv, cv_acc


def run(name, sigs, y, blocks):
    y = np.asarray(y)
    X, _ = tsfresh_featurize_batch(sigs, kind="efficient")
    X = np.nan_to_num(X)
    sel = SelectFdr(f_classif, alpha=0.05).fit(X, y)
    Xs = sel.transform(X)
    cv, groups = make_cv(blocks)
    acc, sd = cv_acc(Xs, y, cv, groups)
    print(f"{name}: tsfresh(efficient) FDR {Xs.shape[1]}/{X.shape[1]} feats -> acc {acc:.3f} +/- {sd:.3f}", flush=True)
    return {"domain": name, "tsfresh_acc": round(acc, 4), "tsfresh_sd": round(sd, 4),
            "kept": int(Xs.shape[1]), "total": int(X.shape[1])}


def main():
    res = []
    sm, ym = memory_demo.build_signals()
    res.append(run("Memory (real)", sm, ym, None))
    sg, yg, bg = graphs_demo.build_signals_blocked()
    res.append(run("Dynamic graphs", sg, yg, bg))
    samples = ecology_demo.load_samples()
    se, ye, be = ecology_demo.build_signals_blocked(samples)
    res.append(run("Ecology (EcoMon)", se, ye, be))
    json.dump(res, open(os.path.join(HERE, "results_tsfresh.json"), "w"), indent=2)
    print("saved results_tsfresh.json", flush=True)


if __name__ == "__main__":
    main()
