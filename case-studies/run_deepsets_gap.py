"""Why is Deep Sets nominally ahead on memory, and is the edge from temporal order?

Deep Sets is order-invariant (sum pooling), so any edge cannot come from temporal
order. The remaining hypothesis is feature expressiveness: a learned nonlinear
pooling versus a small fixed feature set. This script compares, on the real memory
data under a single shared five-fold split, generic compact Psi + RF, a richer fixed
featurization (tsfresh) + RF, and a Deep Sets network. Out-of-fold predictions give
paired bootstrap 95% CIs on the differences (instance-level here, since memory
episodes are independent draws, not autocorrelated blocks).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn"]:
    sys.path.insert(0, os.path.join(HERE, d))
import real_memory_demo as memory_demo
from pipeline import featurize_signals, tsfresh_featurize_batch

SEED = 0
EPOCHS, BATCH = 300, 32
BOOT = 2000


def rf_oof(X, y, splits):
    oof = np.empty(len(y), dtype=object)
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return (oof == y).astype(float)


def deepsets_oof(sigs, yi, C, splits):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
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
        tri = np.asarray(tr)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(tri), generator=g)
            for i in range(0, len(tri), BATCH):
                idx = torch.tensor(tri[perm[i:i + BATCH].numpy()])
                opt.zero_grad()
                lossf(net(Xt[idx]), yt[idx]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            oof[te] = net(Xt[te]).argmax(1).numpy()
    return oof


def boot_ci(c, rng):
    n = len(c)
    s = c[rng.integers(0, n, (BOOT, n))].mean(1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def boot_ci_diff(a, b, rng):
    n = len(a)
    idx = rng.integers(0, n, (BOOT, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    print("building real memory signals...", flush=True)
    sigs, y = memory_demo.build_signals(seed=SEED)
    y = np.asarray(y)
    le = LabelEncoder().fit(y); yi = le.transform(y); C = len(le.classes_)
    X_ours = np.array([featurize_signals(s) for s in sigs])
    print("tsfresh featurization (richer fixed set)...", flush=True)
    X_rich, _ = tsfresh_featurize_batch(sigs, kind="efficient")
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_ours, y))

    c_ours = rf_oof(X_ours, y, splits)
    c_rich = rf_oof(X_rich, y, splits)
    pds = deepsets_oof(sigs, yi, C, splits)
    c_ds = (le.inverse_transform(pds) == y).astype(float)
    rng = np.random.default_rng(SEED + 5)

    out = {"domain": "memory (real)", "n": int(len(y)),
           "ours_dim": int(X_ours.shape[1]), "rich_dim": int(X_rich.shape[1]),
           "acc": {"ours": [round(float(c_ours.mean()), 4), boot_ci(c_ours, rng)],
                   "ours_rich_tsfresh": [round(float(c_rich.mean()), 4), boot_ci(c_rich, rng)],
                   "deepsets": [round(float(c_ds.mean()), 4), boot_ci(c_ds, rng)]},
           "paired_gaps": {}}
    for name, a, b in [("deepsets_minus_ours", c_ds, c_ours),
                       ("deepsets_minus_rich", c_ds, c_rich),
                       ("rich_minus_ours", c_rich, c_ours)]:
        ci = boot_ci_diff(a, b, rng)
        out["paired_gaps"][name] = {"delta": round(float((a - b).mean()), 4), "ci95": ci,
                                    "sig": bool(ci[0] > 0 or ci[1] < 0)}
    print("\n=== Deep Sets gap (memory, real; OOF paired bootstrap) ===")
    for k, v in out["acc"].items():
        print(f"  {k:20s}: {v[0]:.3f}  CI {v[1]}")
    for k, v in out["paired_gaps"].items():
        print(f"  {k:22s}: {v['delta']:+.3f}  CI {v['ci95']}  sig={v['sig']}")
    json.dump(out, open(os.path.join(HERE, "results_deepsets_gap.json"), "w"), indent=2)
    print("saved results_deepsets_gap.json")


if __name__ == "__main__":
    main()
