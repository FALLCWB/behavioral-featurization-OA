"""Statistical validation of the cost/accuracy comparison (ours vs Deep Sets).

Repeats the head-to-head R times per (method, domain), varying the held-out fold and
the model seed, and reports each metric as mean with a 95% confidence interval:
test accuracy, training wall-time, and CPU time. RAM is reported separately from the
isolated single-process measurements (it is deterministic: model size + library
footprint, not a random quantity). This puts the "~10-35x less CPU" claim, and the
small accuracy gap, on a 95%-confidence footing.
"""
from __future__ import annotations
import os
# Single-thread everything so the CPU-time comparison is a clean per-core measurement
# (RandomForest is single-threaded here; torch must match, or the CPU ratio mixes
# algorithmic cost with thread count). Must be set before numpy/torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import time
import resource
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

from pipeline import featurize_signals
R = 20
EPOCHS = 300
BATCH = 32


def _fit_minibatch(net, Xtr, ytr, seed):
    """Fair training for the learned baselines: minibatch SGD (Adam) with gradient
    clipping, the standard setup, so the order-aware net is not handicapped by a
    full-batch regime. Same recipe for Deep Sets and the LSTM."""
    import torch
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    n = len(ytr)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def load(domain):
    import real_memory_demo as memory_demo, graphs_demo, ecology_demo
    if domain == "memory":
        s, y = memory_demo.build_signals()
        return s, np.asarray(y), None
    if domain == "graphs":
        s, y, b = graphs_demo.build_signals_blocked()
        return s, np.asarray(y), b
    s, y, b = ecology_demo.build_signals_blocked(ecology_demo.load_samples())
    return s, np.asarray(y), b


def split(y, blocks, r):
    if blocks is None:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=r)
        return next(sss.split(np.zeros(len(y)), y))
    b = sorted(set(blocks.tolist()))[r % len(set(blocks.tolist()))]
    te = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]
    return tr, te


def cpu():
    g = resource.getrusage(resource.RUSAGE_SELF)
    return g.ru_utime + g.ru_stime


def ci95(a):
    a = np.asarray(a, float)
    m = a.mean()
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return round(float(m), 3), round(float(h), 3)


def run_ours(sigs, y, blocks):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    X = np.array([featurize_signals(s) for s in sigs])
    acc, tr_s, cp_s = [], [], []
    for r in range(R):
        tr, te = split(y, blocks, r)
        c0, t0 = cpu(), time.perf_counter()
        clf = RandomForestClassifier(n_estimators=300, random_state=r, n_jobs=1).fit(X[tr], y[tr])
        tr_s.append(time.perf_counter() - t0)
        cp_s.append(cpu() - c0)
        acc.append(accuracy_score(y[te], clf.predict(X[te])))
    return acc, tr_s, cp_s


def run_deepsets(sigs, y, blocks):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    torch.set_num_threads(1)
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    names = sorted(sigs[0])
    T, J, C = len(sigs[0][names[0]]), len(names), len(le.classes_)
    Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)
    acc, tr_s, cp_s = [], [], []
    for r in range(R):
        torch.manual_seed(r)
        tr, te = split(y, blocks, r)
        mu = Xseq[tr].reshape(-1, J).mean(0)
        sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xt = torch.tensor((Xseq - mu) / sd)
        yt = torch.tensor(yi)


        class DS(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(J, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, C))

            def forward(self, x):
                return self.rho(self.phi(x).sum(1))

        net = DS()
        c0, t0 = cpu(), time.perf_counter()
        _fit_minibatch(net, Xt[tr], yt[tr], r)
        tr_s.append(time.perf_counter() - t0)
        cp_s.append(cpu() - c0)
        net.eval()
        with torch.no_grad():
            acc.append(float((net(Xt[te]).argmax(1).numpy() == yi[te]).mean()))
    return acc, tr_s, cp_s


def run_lstm(sigs, y, blocks):
    """ORDER-AWARE learned baseline: a recurrent network (LSTM) reads the signal
    sequence in time order and classifies from the final hidden state. Unlike Deep
    Sets (sum-pooling, order-invariant), this model can exploit temporal order; it
    is the order-aware learned counterpart for the cost comparison."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    torch.set_num_threads(1)
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    names = sorted(sigs[0])
    T, J, C = len(sigs[0][names[0]]), len(names), len(le.classes_)
    Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)
    acc, tr_s, cp_s = [], [], []
    for r in range(R):
        torch.manual_seed(r)
        tr, te = split(y, blocks, r)
        mu = Xseq[tr].reshape(-1, J).mean(0)
        sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xt = torch.tensor((Xseq - mu) / sd)
        yt = torch.tensor(yi)

        class LSTMClf(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(J, 64, batch_first=True)
                self.head = nn.Linear(64, C)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.head(h[-1])

        net = LSTMClf()
        c0, t0 = cpu(), time.perf_counter()
        _fit_minibatch(net, Xt[tr], yt[tr], r)
        tr_s.append(time.perf_counter() - t0)
        cp_s.append(cpu() - c0)
        net.eval()
        with torch.no_grad():
            acc.append(float((net(Xt[te]).argmax(1).numpy() == yi[te]).mean()))
    return acc, tr_s, cp_s


def main():
    out = []
    for domain in ["memory", "graphs", "ecology"]:
        sigs, y, blocks = load(domain)
        print(f"\n### {domain} (R={R} repeats, 95% CI)", flush=True)
        rec = {"domain": domain}
        for mname, fn in [("ours", run_ours), ("deepsets", run_deepsets), ("lstm", run_lstm)]:
            acc, tr_s, cp_s = fn(sigs, y, blocks)
            rec[mname] = {"acc": ci95(acc), "train_s": ci95(tr_s), "cpu_s": ci95(cp_s)}
            am, ah = rec[mname]["acc"]
            tm, th = rec[mname]["train_s"]
            cm, ch = rec[mname]["cpu_s"]
            print(f"  {mname:9s}: acc {am:.3f}+/-{ah:.3f}  train {tm:.2f}+/-{th:.2f}s  "
                  f"cpu {cm:.2f}+/-{ch:.2f}s", flush=True)
        # mean CPU ratio of each learned baseline over ours (paired splits per repeat)
        rec["cpu_ratio_ds_over_ours"] = round(rec["deepsets"]["cpu_s"][0] / max(rec["ours"]["cpu_s"][0], 1e-9), 1)
        rec["cpu_ratio_lstm_over_ours"] = round(rec["lstm"]["cpu_s"][0] / max(rec["ours"]["cpu_s"][0], 1e-9), 1)
        print(f"  -> CPU ratio (DeepSets / ours): {rec['cpu_ratio_ds_over_ours']}x"
              f" | (LSTM / ours): {rec['cpu_ratio_lstm_over_ours']}x", flush=True)
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, "results_resource_stats.json"), "w"), indent=2)
    print("\nsaved results_resource_stats.json", flush=True)


if __name__ == "__main__":
    main()
