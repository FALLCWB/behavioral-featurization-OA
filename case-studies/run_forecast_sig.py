"""Significance of the order advantage on ecology forecasting (the conjunction case:
an evolving, volatile-dimensionality object with a natural dynamical label).

Paired bootstrap 95% CI of (behavioral - set_pool) and (behavioral - behavioral_shuffled)
accuracy, using out-of-fold predictions under the leakage-controlled blocked CV.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import run_forecasting as F
from pipeline import featurize_signals, pool_features, shuffle_signals

SEED = 0


def oof_correct(sigs, y, block, fn):
    X = np.array([fn(s) for s in sigs])
    cv = StratifiedGroupKFold(5)
    pred = cross_val_predict(RandomForestClassifier(n_estimators=400, random_state=SEED),
                             X, y, cv=cv, groups=block, n_jobs=-1)
    return (pred == y).astype(float)


def boot_ci(a, b, block, n_boot=2000):
    """Cluster (block) bootstrap: resample whole time blocks with replacement, not
    individual windows. Windows inside a block are temporally autocorrelated, so an
    instance-level bootstrap would treat them as i.i.d. and understate the interval.
    Resampling blocks respects the dependence structure of the leakage-controlled CV."""
    rng = np.random.default_rng(SEED + 7)
    block = np.asarray(block)
    groups = [np.where(block == u)[0] for u in np.unique(block)]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[c] for c in chosen])
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float((a.mean() - b.mean())), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    sigs, y, block = F.make_forecast(F.ecology_sources(), "richness")
    cb = oof_correct(sigs, y, block, featurize_signals)
    cp = oof_correct(sigs, y, block, pool_features)
    cs = oof_correct(sigs, y, block, lambda s: featurize_signals(shuffle_signals(s, rng)))
    print(f"ecology forecasting (n={len(y)}): behavioral={cb.mean():.3f} pool={cp.mean():.3f} shuffled={cs.mean():.3f}")
    out = {"n": int(len(y)),
           "behavioral": round(float(cb.mean()), 4), "pool": round(float(cp.mean()), 4),
           "shuffled": round(float(cs.mean()), 4), "diffs": {}}
    for name, key, a, b in [("behavioral - pool", "behavioral_minus_pool", cb, cp),
                            ("behavioral - shuffled", "behavioral_minus_shuffled", cb, cs)]:
        d, lo, hi = boot_ci(a, b, block)
        sig = bool(lo > 0 or hi < 0)
        print(f"  {name}: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> "
              f"{'SIGNIFICANT' if sig else 'not significant (CI crosses 0)'}")
        out["diffs"][key] = {"delta": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "sig": sig}
    json.dump(out, open(os.path.join(HERE, "results_forecast_sig.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
