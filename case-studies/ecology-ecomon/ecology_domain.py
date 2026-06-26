"""Domain-informed ecological community features (per Filipe's A9: features grounded
in each domain's literature, not a generic pool).

Per community sample, abundance-weighted diversity indices (Magurran 2004):
richness S, log total abundance, Shannon H, Gini-Simpson (1-lambda), Pielou evenness
J, Berger-Parker dominance. Between consecutive samples (Whittaker 1972 turnover):
Bray-Curtis dissimilarity (abundance-weighted), Jaccard turnover (presence), and the
fraction of newly seen taxa. These nine signals over a window are then featurized by
the same compact Psi and used to classify the region, under the leakage-controlled
blocked CV. Question: does this beat the generic 0.634 baseline?
"""
from __future__ import annotations
import os
import sys
import csv
import math
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "case-studies"))
from pipeline import featurize_signals
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))

CSV = os.path.join(HERE, "data", "ecomon.csv")
WIN = 20
TARGET_PER_REGION = 300


def region_of(lat):
    if lat < 39.5:
        return "south"
    if lat < 42.0:
        return "mid"
    return "north"


def load_samples_abund():
    """Per sample: date, lat, and the taxon->abundance dict (for full diversity stats)."""
    taxa_id = {}
    samples = {}
    with open(CSV, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col = {n: i for i, n in enumerate(header)}
        ci_date, ci_cr, ci_st = col["date"], col["cruiseid"], col["station"]
        ci_lat, ci_taxon, ci_count = col["lat"], col["taxon"], col["count"]
        for row in r:
            cnt = row[ci_count]
            if cnt == "nd":
                continue
            try:
                c = float(cnt)
            except ValueError:
                continue
            if c <= 0:
                continue
            key = (row[ci_date], row[ci_cr], row[ci_st])
            s = samples.get(key)
            if s is None:
                try:
                    lat = float(row[ci_lat])
                except ValueError:
                    continue
                s = {"date": row[ci_date], "lat": lat, "ab": {}}
                samples[key] = s
            tx = row[ci_taxon]
            tid = taxa_id.get(tx)
            if tid is None:
                tid = len(taxa_id)
                taxa_id[tx] = tid
            s["ab"][tid] = s["ab"].get(tid, 0.0) + c
    return list(samples.values())


def indices(ab):
    n = np.array(list(ab.values()), dtype=float)
    N = n.sum()
    S = len(n)
    p = n / N
    H = float(-(p * np.log(p)).sum())
    simpson = float((p * p).sum())
    return {
        "richness": float(S),
        "log_abund": math.log1p(N),
        "shannon": H,
        "gini_simpson": 1.0 - simpson,
        "evenness": (H / math.log(S)) if S > 1 else 0.0,
        "dominance": float(n.max() / N),
    }


def bray_curtis(a, b):
    keys = set(a) | set(b)
    num = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
    den = sum(a.get(k, 0.0) + b.get(k, 0.0) for k in keys)
    return num / den if den > 0 else 0.0


def window_signals(win):
    sig = {k: [] for k in ["richness", "log_abund", "shannon", "gini_simpson",
                           "evenness", "dominance", "bray_curtis", "jaccard", "new_taxa_rate"]}
    seen = set()
    prev = None
    for s in win:
        ab = s["ab"]
        idx = indices(ab)
        for k in ["richness", "log_abund", "shannon", "gini_simpson", "evenness", "dominance"]:
            sig[k].append(idx[k])
        cur = set(ab)
        if prev is not None:
            sig["bray_curtis"].append(bray_curtis(prev_ab, ab))
            sig["jaccard"].append(1.0 - len(cur & prev) / max(1, len(cur | prev)))
        else:
            sig["bray_curtis"].append(0.0)
            sig["jaccard"].append(0.0)
        sig["new_taxa_rate"].append(len(cur - seen) / max(1, len(cur)))
        seen |= cur
        prev = cur
        prev_ab = ab
    return sig


def build(samples, n_blocks=5):
    by = {"south": [], "mid": [], "north": []}
    for s in samples:
        by[region_of(s["lat"])].append(s)
    X, y, block = [], [], []
    for region, lst in by.items():
        lst.sort(key=lambda s: s["date"])
        nwin = len(lst) // WIN
        for i in range(nwin):
            X.append(featurize_signals(window_signals(lst[i * WIN:(i + 1) * WIN])))
            y.append(region)
            block.append(min(n_blocks - 1, i * n_blocks // max(nwin, 1)))
    return np.array(X), np.array(y), np.array(block)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    print("loading abundance samples...", flush=True)
    samples = load_samples_abund()
    X, y, block = build(samples)
    m = RandomForestClassifier(n_estimators=300, random_state=0)
    s = cross_val_score(m, X, y, cv=StratifiedGroupKFold(5), groups=block, n_jobs=-1)
    print(f"Ecology with DOMAIN-INFORMED features: n={len(y)}, {X.shape[1]} features", flush=True)
    print(f"  accuracy = {s.mean():.3f} +/- {s.std():.3f}   (generic baseline was 0.634)", flush=True)
    json.dump({"acc": round(float(s.mean()), 4), "sd": round(float(s.std()), 4),
               "n": int(len(y)), "features": int(X.shape[1]), "baseline": 0.634},
              open(os.path.join(os.path.dirname(HERE), "results_ecology_domain.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
