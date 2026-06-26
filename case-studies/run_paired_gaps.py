"""Significance-backed (95% CI) paired comparison of the compact featurization (ours)
against every learned baseline, on each domain, under the SAME leakage-controlled
out-of-fold protocol the paper already uses (run_deepsets_gap / run_ecology_gap):

  splits  : StratifiedKFold(5) for memory (independent episodes),
            StratifiedGroupKFold(5) by time-block for graphs and ecology.
  OOF     : one held-out prediction per window, pooled across folds.
  gap     : paired bootstrap 95% CI on (baseline_correct - ours_correct), instance
            level, with a significance flag (CI excludes 0).

Baselines: deepsets, lstm, mil, settransformer, tcn (all on the (N,T,J) frame), and
dyngraph (temporal GNN, graphs only). This makes every "ours wins / ties / loses"
claim a 95%-CI statement rather than an eyeball of overlapping marginal intervals.

    .venv/bin/python run_paired_gaps.py --domains graphs --methods mil --folds 5
    .venv/bin/python run_paired_gaps.py                      # all, full
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

from run_resource_stats import load, EPOCHS, BATCH
from run_resource_extra import _prep, _make_model
from pipeline import featurize_signals

SEED = 0
BOOT = 2000


def make_splits(y, blocks, k=5):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    if blocks is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    return list(StratifiedGroupKFold(k).split(np.zeros(len(y)), y, groups=blocks))


def rf_oof(sigs, y, splits):
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([featurize_signals(s) for s in sigs])
    oof = np.empty(len(y), dtype=object)
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return (oof == y).astype(float)


def learned_oof(method, sigs, y, splits):
    import torch
    torch.set_num_threads(1)
    Xseq, yi, J, C = _prep(sigs, y)
    yt = torch.tensor(yi)
    oof = np.zeros(len(yi), dtype=int)
    for r, (tr, te) in enumerate(splits):
        torch.manual_seed(r)
        tri = np.asarray(tr)
        mu = Xseq[tri].reshape(-1, J).mean(0); sd = Xseq[tri].reshape(-1, J).std(0) + 1e-6
        Xt = torch.tensor((Xseq - mu) / sd)
        net = _make_model(method, J, C)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(r)
        net.train()
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
    return (oof == yi).astype(float)


def dyngraph_oof(splits_blocks):
    """Temporal-GNN OOF on the graph windows, on the SAME block-group folds."""
    import torch
    import run_resource_dyngraph as DG
    from sklearn.preprocessing import LabelEncoder
    torch.set_num_threads(1)
    windows, y, blocks = DG.build_graph_windows()
    le = LabelEncoder().fit(y); yi = le.transform(y); C = len(le.classes_)
    prepped = DG.precompute(windows)
    splits = make_splits(y, blocks)         # same StratifiedGroupKFold by block
    oof = np.zeros(len(yi), dtype=int)
    for r, (tr, te) in enumerate(splits):
        torch.manual_seed(r)
        net = DG.make_model(C)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(r)
        tri = np.asarray(tr)
        net.train()
        for _ in range(EPOCHS):
            perm = tri[torch.randperm(len(tri), generator=g).numpy()]
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                A, X, ws, B = DG.combine([prepped[j] for j in idx])
                lossf(net.forward_batch(A, X, ws, B), torch.tensor(yi[idx])).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            A, X, ws, B = DG.combine([prepped[j] for j in te])
            oof[te] = net.forward_batch(A, X, ws, B).argmax(1).numpy()
    return (oof == yi).astype(float), y


def boot_ci(c, rng):
    n = len(c); s = c[rng.integers(0, n, (BOOT, n))].mean(1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def boot_ci_diff(a, b, rng):
    n = len(a); idx = rng.integers(0, n, (BOOT, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="memory,graphs,ecology")
    ap.add_argument("--methods", default="deepsets,lstm,mil,settransformer,tcn")
    ap.add_argument("--out", default="results_paired_gaps.json")
    args = ap.parse_args()
    methods = args.methods.split(",")
    rng = np.random.default_rng(SEED + 5)
    out = []
    for domain in args.domains.split(","):
        sigs, y, blocks = load(domain)
        splits = make_splits(y, blocks)
        c_ours = rf_oof(sigs, y, splits)
        rec = {"domain": domain, "n": int(len(y)),
               "ours_acc": [round(float(c_ours.mean()), 4), boot_ci(c_ours, rng)], "gaps": {}}
        print(f"\n### {domain}  ours acc {rec['ours_acc'][0]:.3f} CI {rec['ours_acc'][1]}", flush=True)
        for m in methods:
            if m == "dyngraph":
                continue  # graphs-only, handled by the special branch below
            c_b = learned_oof(m, sigs, y, splits)
            ci = boot_ci_diff(c_b, c_ours, rng)
            rec["gaps"][m] = {"baseline_acc": [round(float(c_b.mean()), 4), boot_ci(c_b, rng)],
                              "delta_baseline_minus_ours": round(float((c_b - c_ours).mean()), 4),
                              "ci95": ci, "sig": bool(ci[0] > 0 or ci[1] < 0)}
            g = rec["gaps"][m]
            print(f"  {m:14s}: acc {g['baseline_acc'][0]:.3f}  gap(base-ours) "
                  f"{g['delta_baseline_minus_ours']:+.3f} CI {ci} sig={g['sig']}", flush=True)
        if domain == "graphs" and "dyngraph" in args.methods.split(",") + ["dyngraph"]:
            c_dg, _ = dyngraph_oof(splits)
            ci = boot_ci_diff(c_dg, c_ours, rng)
            rec["gaps"]["dyngraph"] = {"baseline_acc": [round(float(c_dg.mean()), 4), boot_ci(c_dg, rng)],
                                       "delta_baseline_minus_ours": round(float((c_dg - c_ours).mean()), 4),
                                       "ci95": ci, "sig": bool(ci[0] > 0 or ci[1] < 0)}
            g = rec["gaps"]["dyngraph"]
            print(f"  {'dyngraph':14s}: acc {g['baseline_acc'][0]:.3f}  gap(base-ours) "
                  f"{g['delta_baseline_minus_ours']:+.3f} CI {ci} sig={g['sig']}", flush=True)
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, args.out), "w"), indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
