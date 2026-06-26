"""P2: does a richer FIXED featurization close the Deep Sets edge on ecology, on the
SAME splits? Compares, window-for-window under one leakage-controlled blocked CV:
generic compact Psi + RF, domain-informed community indices + RF, and a Deep Sets
network on the generic signals. Out-of-fold predictions give paired bootstrap 95%
CIs on the differences, so ``domain features close/exceed the Deep Sets gap'' is a
within-experiment claim, not a cross-harness comparison.

The generic and domain windows are aligned by construction: both cut non-overlapping
WIN=20 windows per region, sorted by date, from the same EcoMon samples.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
import ecology_demo as ECO
import ecology_domain as DOM
from pipeline import featurize_signals

SEED = 0
EPOCHS, BATCH = 300, 32
BOOT = 2000


def _deepsets_oof(sigs, yi, C, splits):
    import torch
    import torch.nn as nn
    torch.set_num_threads(8)
    names = sorted(sigs[0])
    J = len(names)
    Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)
    oof = np.zeros(len(yi), dtype=int)
    yt = torch.tensor(yi)
    for r, (tr, te) in enumerate(splits):
        torch.manual_seed(r)
        mu = Xseq[tr].reshape(-1, J).mean(0); sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xt = torch.tensor((Xseq - mu) / sd)

        class DS(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(J, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, C))

            def forward(self, x):
                return self.rho(self.phi(x).sum(1))

        net = DS()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(r)
        net.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(tr), generator=g)
            for i in range(0, len(tr), BATCH):
                idx = torch.tensor(tr)[perm[i:i + BATCH]]
                opt.zero_grad()
                lossf(net(Xt[idx]), yt[idx]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            oof[te] = net(Xt[te]).argmax(1).numpy()
    return oof


def _rf_oof(X, y, splits):
    oof = np.empty(len(y), dtype=object)
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof


def boot_ci(c, rng):
    n = len(c)
    return [round(float(np.percentile(c[rng.integers(0, n, (BOOT, n))].mean(1), p)), 4)
            for p in (2.5, 97.5)]


def boot_ci_diff(a, b, rng):
    n = len(a)
    idx = rng.integers(0, n, (BOOT, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    sg, yg, bg = ECO.build_signals_blocked(ECO.load_samples())
    Xd, yd, bd = DOM.build(DOM.load_samples_abund())
    assert list(yg) == list(yd) and list(bg) == list(bd), "generic/domain windows misaligned"
    y, blocks = yg, bg
    Xg = np.array([featurize_signals(s) for s in sg])
    le = LabelEncoder().fit(y); yi = le.transform(y); C = len(le.classes_)
    splits = list(StratifiedGroupKFold(5).split(Xg, y, groups=blocks))

    pg = _rf_oof(Xg, y, splits)
    pd_ = _rf_oof(Xd, y, splits)
    pds = _deepsets_oof(sg, yi, C, splits)
    cg = (pg == y).astype(float)
    cd = (pd_ == y).astype(float)
    cs = (le.inverse_transform(pds) == y).astype(float)
    rng = np.random.default_rng(SEED)

    out = {"n_windows": int(len(y)), "n_blocks": int(len(set(blocks.tolist()))),
           "generic_dim": int(Xg.shape[1]), "domain_dim": int(Xd.shape[1]),
           "acc": {"generic": [round(float(cg.mean()), 4), boot_ci(cg, rng)],
                   "domain":  [round(float(cd.mean()), 4), boot_ci(cd, rng)],
                   "deepsets":[round(float(cs.mean()), 4), boot_ci(cs, rng)]},
           "paired_gaps": {
               "domain_minus_generic": {"delta": round(float((cd - cg).mean()), 4),
                                        "ci95": boot_ci_diff(cd, cg, rng)},
               "deepsets_minus_generic": {"delta": round(float((cs - cg).mean()), 4),
                                          "ci95": boot_ci_diff(cs, cg, rng)},
               "domain_minus_deepsets": {"delta": round(float((cd - cs).mean()), 4),
                                         "ci95": boot_ci_diff(cd, cs, rng)}}}
    for k, v in out["paired_gaps"].items():
        v["sig"] = bool(v["ci95"][0] > 0 or v["ci95"][1] < 0)
    print("=== Ecology: generic vs domain vs Deep Sets, SAME blocked-CV splits ===")
    for k, v in out["acc"].items():
        print(f"  {k:9s}: {v[0]:.3f}  95% CI {v[1]}")
    for k, v in out["paired_gaps"].items():
        print(f"  {k:24s}: {v['delta']:+.3f}  CI {v['ci95']}  sig={v['sig']}")
    json.dump(out, open(os.path.join(HERE, "results_ecology_gap.json"), "w"), indent=2)
    print("saved results_ecology_gap.json")


if __name__ == "__main__":
    main()
