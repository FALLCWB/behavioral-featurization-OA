"""Resource head-to-head: our pipeline (compact featurization + RandomForest) vs a
Deep Sets network (learns directly on the variable-size object, end to end).

Run as a worker for ONE (method, domain) so peak RAM is measured per process:
    python run_resource.py --method {ours,deepsets} --domain {memory,graphs,ecology}
Prints one JSON line with accuracy, train wall-time, CPU time, and peak RSS.
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import resource
import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)


def load(domain):
    import real_memory_demo as memory_demo, graphs_demo, ecology_demo
    if domain == "memory":
        sigs, y = memory_demo.build_signals()
        blocks = None
    elif domain == "graphs":
        sigs, y, blocks = graphs_demo.build_signals_blocked()
    else:
        sigs, y, blocks = ecology_demo.build_signals_blocked(ecology_demo.load_samples())
    return sigs, np.asarray(y), blocks


def split(y, blocks):
    if blocks is None:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.25, random_state=0, stratify=y)
        return tr, te
    te = np.where(blocks == blocks.max())[0]
    tr = np.where(blocks != blocks.max())[0]
    return tr, te


def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on Linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--domain", required=True)
    args = ap.parse_args()
    proc = psutil.Process()
    sigs, y, blocks = load(args.domain)
    tr, te = split(y, blocks)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    out = {"method": args.method, "domain": args.domain, "n": int(len(y)),
           "n_train": int(len(tr)), "n_test": int(len(te))}

    if args.method == "ours":
        from pipeline import featurize_signals
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        X = np.array([featurize_signals(s) for s in sigs])
        out["repr_dim"] = int(X.shape[1])
        base = proc.memory_info().rss / 1024**2
        c0, t0 = cpu_now(), time.perf_counter()
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1).fit(X[tr], y[tr])
        out["train_s"] = round(time.perf_counter() - t0, 2)
        out["cpu_s"] = round(cpu_now() - c0, 2)
        out["test_acc"] = round(float(accuracy_score(y[te], clf.predict(X[te]))), 3)
        out["params"] = None
        out["needs_gpu"] = False
    elif args.method == "deepsets":
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        torch.set_num_threads(1)
        names = sorted(sigs[0])
        T = len(sigs[0][names[0]])
        J = len(names)
        Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)  # (N,T,J)
        mu = Xseq[tr].reshape(-1, J).mean(0)
        sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xseq = (Xseq - mu) / sd
        C = len(le.classes_)
        Xt = torch.tensor(Xseq)
        yt = torch.tensor(yi)

        class DeepSets(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(J, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, C))

            def forward(self, x):
                return self.rho(self.phi(x).sum(1))

        net = DeepSets()
        out["params"] = int(sum(p.numel() for p in net.parameters()))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xtr, ytr_ = Xt[tr], yt[tr]
        base = proc.memory_info().rss / 1024**2
        c0, t0 = cpu_now(), time.perf_counter()
        net.train()
        g = torch.Generator().manual_seed(0)
        ntr = Xtr.shape[0]
        for _ in range(300):
            perm = torch.randperm(ntr, generator=g)
            for i in range(0, ntr, 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = lossf(net(Xtr[idx]), ytr_[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        out["train_s"] = round(time.perf_counter() - t0, 2)
        out["cpu_s"] = round(cpu_now() - c0, 2)
        net.eval()
        with torch.no_grad():
            pred = net(Xt[te]).argmax(1).numpy()
        out["test_acc"] = round(float((pred == yi[te]).mean()), 3)
        out["repr_dim"] = None
        out["needs_gpu"] = False  # trains on CPU at this scale; GPU only helps at larger scale
    else:  # lstm: order-aware learned baseline (recurrent)
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        torch.set_num_threads(1)
        names = sorted(sigs[0])
        T = len(sigs[0][names[0]])
        J = len(names)
        Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)
        mu = Xseq[tr].reshape(-1, J).mean(0)
        sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xseq = (Xseq - mu) / sd
        C = len(le.classes_)
        Xt = torch.tensor(Xseq)
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
        out["params"] = int(sum(p.numel() for p in net.parameters()))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xtr, ytr_ = Xt[tr], yt[tr]
        base = proc.memory_info().rss / 1024**2
        c0, t0 = cpu_now(), time.perf_counter()
        net.train()
        g = torch.Generator().manual_seed(0)
        ntr = Xtr.shape[0]
        for _ in range(300):
            perm = torch.randperm(ntr, generator=g)
            for i in range(0, ntr, 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = lossf(net(Xtr[idx]), ytr_[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        out["train_s"] = round(time.perf_counter() - t0, 2)
        out["cpu_s"] = round(cpu_now() - c0, 2)
        net.eval()
        with torch.no_grad():
            pred = net(Xt[te]).argmax(1).numpy()
        out["test_acc"] = round(float((pred == yi[te]).mean()), 3)
        out["repr_dim"] = None
        out["needs_gpu"] = False  # trains on CPU at this scale; GPU only helps at larger scale

    out["rss_after_setup_mb"] = round(base, 1)
    out["peak_rss_mb"] = round(peak_rss_mb(), 1)
    out["model_rss_mb"] = round(out["peak_rss_mb"] - base, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
