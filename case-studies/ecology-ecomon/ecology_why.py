"""Why does ecology region-classification cap at ~0.63, regardless of features?

Clean, balanced comparison on the SAME windows (300 evenly-spaced per region):
  - generic behavioral features vs domain-informed ecological features (does richer
    feature engineering move the number?);
  - region confusion matrix (where does the error go -> do the lat-band regions
    overlap?);
  - season classification with the same features (is the community governed by
    season instead, i.e. are the features fine but region a weak target?).
"""
from __future__ import annotations
import os
import sys
import math
import json
import numpy as np
from statistics import median
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import ecology_domain as ED
from pipeline import featurize_signals

PER_REGION = 300


def season_of(m):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def generic_window(win):
    """The original generic behavioral signals, derived from abundance dicts."""
    sig = {k: [] for k in ["richness", "log_abund", "shannon", "turnover", "new_taxa_rate"]}
    seen, prev = set(), None
    for s in win:
        ab = s["ab"]
        n = np.array(list(ab.values()), dtype=float)
        N = n.sum()
        p = n / N
        cur = set(ab)
        sig["richness"].append(float(len(ab)))
        sig["log_abund"].append(math.log1p(N))
        sig["shannon"].append(float(-(p * np.log(p)).sum()))
        sig["turnover"].append(float(len(cur ^ prev)) if prev is not None else 0.0)
        sig["new_taxa_rate"].append(len(cur - seen) / max(1, len(cur)))
        seen |= cur
        prev = cur
    return sig


def build():
    samples = ED.load_samples_abund()
    by = {"south": [], "mid": [], "north": []}
    for s in samples:
        by[ED.region_of(s["lat"])].append(s)
    Xg, Xd, yreg, ysea, block = [], [], [], [], []
    for region, lst in by.items():
        lst.sort(key=lambda s: s["date"])
        nwin = len(lst) // ED.WIN
        take = np.unique(np.linspace(0, nwin - 1, min(PER_REGION, nwin)).astype(int))
        for j, i in enumerate(take):
            w = lst[i * ED.WIN:(i + 1) * ED.WIN]
            Xg.append(featurize_signals(generic_window(w)))
            Xd.append(featurize_signals(ED.window_signals(w)))
            yreg.append(region)
            months = [int(s["date"][5:7]) for s in w]
            ysea.append(season_of(int(round(median(months)))))
            block.append(min(4, j * 5 // max(len(take), 1)))
    return (np.array(Xg), np.array(Xd), np.array(yreg), np.array(ysea), np.array(block))


def acc(X, y, block):
    s = cross_val_score(RandomForestClassifier(300, random_state=0), X, y,
                        cv=StratifiedGroupKFold(5), groups=block, n_jobs=-1)
    return float(s.mean()), float(s.std())


def main():
    print("loading...", flush=True)
    Xg, Xd, yreg, ysea, block = build()
    n = len(yreg)
    dist = {r: int((yreg == r).sum()) for r in sorted(set(yreg.tolist()))}
    print(f"n={n} (balanced): {dist}", flush=True)
    ga, gsd = acc(Xg, yreg, block)
    da, dsd = acc(Xd, yreg, block)
    sa, ssd = acc(Xd, ysea, block)
    print(f"  REGION, generic features:        {ga:.3f} +/- {gsd:.3f}", flush=True)
    print(f"  REGION, domain-informed features: {da:.3f} +/- {dsd:.3f}", flush=True)
    print(f"  SEASON, domain-informed features: {sa:.3f} +/- {ssd:.3f}  (chance 0.25)", flush=True)
    pred = cross_val_predict(RandomForestClassifier(300, random_state=0), Xd, yreg,
                             cv=StratifiedGroupKFold(5), groups=block, n_jobs=-1)
    labels = sorted(set(yreg.tolist()))
    cm = confusion_matrix(yreg, pred, labels=labels)
    print(f"  region confusion (rows=true {labels}):", flush=True)
    for lab, row in zip(labels, cm):
        print(f"    {lab:6s}: {row.tolist()}", flush=True)
    json.dump({"region_generic": round(ga, 4), "region_domain": round(da, 4),
               "season_domain": round(sa, 4), "labels": labels, "confusion": cm.tolist(),
               "balanced_n": dist},
              open(os.path.join(os.path.dirname(HERE), "results_ecology_why.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
