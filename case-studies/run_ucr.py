"""Decisive real-data order test: time-series classification where the natural label
IS the shape (temporal order), on standard UCR datasets with real labels.

ECG200 (real heartbeats: normal vs ischemia), GunPoint (motion), ItalyPowerDemand
(daily power demand). Each instance is one univariate series; the class depends on
the temporal SHAPE, so a marginal-only representation (a bag of values: pool/sketch)
should be weak while an order-retaining representation (behavioral functionals with
trend/autocorrelation, or a path signature) should be strong. Uses the official
train/test split (instances are independent; no CV leakage concern).
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from pipeline import (featurize_signals, pool_features, sketch_features,
                      signature_features, shuffle_signals)

DATASETS = ["ECG200", "GunPoint", "ItalyPowerDemand"]
SEED = 0
BOOT = 2000        # paired bootstrap resamples over test instances


def load(ds, split):
    path = os.path.join(HERE, "ucr", "data", ds, f"{ds}_{split}.txt")
    data = np.loadtxt(path)
    return data[:, 1:], data[:, 0].astype(int)


def sd(series):
    return {"x": list(np.asarray(series, dtype=float))}


def ci_from_boot(samples):
    return [round(float(np.percentile(samples, 2.5)), 4),
            round(float(np.percentile(samples, 97.5)), 4)]


def run(ds):
    Xtr, ytr = load(ds, "TRAIN")
    Xte, yte = load(ds, "TEST")
    rng = np.random.default_rng(SEED)
    reps = {
        "behavioral": lambda s: featurize_signals(sd(s)),
        "signature_order": lambda s: signature_features(sd(s)),
        "set_pool": lambda s: pool_features(sd(s)),
        "sketch": lambda s: sketch_features(sd(s)),
        "behavioral_shuffled": lambda s: featurize_signals(shuffle_signals(sd(s), rng)),
    }
    out = {"dataset": ds, "n_train": len(ytr), "n_test": len(yte),
           "series_len": int(Xtr.shape[1]),
           "majority": round(max(np.mean(yte == c) for c in set(yte.tolist())), 4),
           "acc": {}, "acc_ci": {}}
    print(f"\n### {ds} | train={len(ytr)} test={len(yte)} len={Xtr.shape[1]} | majority={out['majority']:.3f}")

    # per-instance correctness vectors (held-out official test split, fixed model)
    correct = {}
    for rn, fn in reps.items():
        Ftr = np.array([fn(x) for x in Xtr])
        Fte = np.array([fn(x) for x in Xte])
        clf = RandomForestClassifier(n_estimators=400, random_state=SEED)
        clf.fit(Ftr, ytr)
        pred = clf.predict(Fte)
        correct[rn] = (pred == yte).astype(float)
        out["acc"][rn] = round(float(accuracy_score(yte, pred)), 4)

    # paired bootstrap over test instances: same resampled indices across reps, so
    # the difference CIs (the order gaps) are paired and account for shared test draws.
    nte = len(yte)
    rng_boot = np.random.default_rng(SEED + 1)   # independent of the shuffle RNG
    boot_idx = rng_boot.integers(0, nte, size=(BOOT, nte))
    boot_acc = {rn: correct[rn][boot_idx].mean(axis=1) for rn in reps}   # (BOOT,)
    for rn in reps:
        out["acc_ci"][rn] = ci_from_boot(boot_acc[rn])
        lo, hi = out["acc_ci"][rn]
        print(f"  {rn:22s}: {out['acc'][rn]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    def two_sided(ci):
        return bool(ci[0] > 0 or ci[1] < 0)

    # order gap (behavioral - shuffled), paired CI, two-sided significance
    gap = boot_acc["behavioral"] - boot_acc["behavioral_shuffled"]
    out["order_gap"] = round(out["acc"]["behavioral"] - out["acc"]["behavioral_shuffled"], 4)
    out["order_gap_ci"] = ci_from_boot(gap)
    out["order_gap_sig"] = two_sided(out["order_gap_ci"])
    # best order-aware vs best marginal-only: pick the winners on the POINT estimate,
    # then bootstrap that fixed pair (so the CI matches the reported statistic, no
    # per-resample-max Jensen inflation).
    best_oa = "behavioral" if out["acc"]["behavioral"] >= out["acc"]["signature_order"] else "signature_order"
    best_mo = "set_pool" if out["acc"]["set_pool"] >= out["acc"]["sketch"] else "sketch"
    out["order_aware_minus_marginal"] = round(out["acc"][best_oa] - out["acc"][best_mo], 4)
    out["order_aware_minus_marginal_ci"] = ci_from_boot(boot_acc[best_oa] - boot_acc[best_mo])
    out["order_aware_minus_marginal_sig"] = two_sided(out["order_aware_minus_marginal_ci"])
    g_lo, g_hi = out["order_gap_ci"]
    d_lo, d_hi = out["order_aware_minus_marginal_ci"]
    print(f"  -> order gap (behavioral - shuffled): {out['order_gap']:+.3f}  95% CI [{g_lo:+.3f}, {g_hi:+.3f}]  sig={out['order_gap_sig']}")
    print(f"  -> best order-aware - best marginal-only: {out['order_aware_minus_marginal']:+.3f}  95% CI [{d_lo:+.3f}, {d_hi:+.3f}]  sig={out['order_aware_minus_marginal_sig']}")
    return out


def main():
    res = [run(ds) for ds in DATASETS]
    json.dump(res, open(os.path.join(HERE, "results_ucr.json"), "w"), indent=2)
    print("\nsaved results_ucr.json")


if __name__ == "__main__":
    main()
