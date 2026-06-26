"""Additional learned baselines for the cost/accuracy comparison, on the SAME
(N, T, J) windowed frame and the SAME fair training protocol as Deep Sets and the
LSTM (run_resource_stats.py). Added in response to reviewer feedback asking the
method to be positioned against closely related learned approaches:

  - mil            : attention-based Multiple-Instance Learning pooling (gated
                     attention, Ilse et al. 2018). ORDER-INVARIANT: the attention
                     weight of a snapshot depends on its content, not its time
                     index, so the pooled representation is a function of the
                     window's time-marginal -> a B_marg instance, like Deep Sets.
  - settransformer : Set Transformer (Lee et al. 2019), SAB self-attention over the
                     window's snapshots + pooling-by-multihead-attention (PMA),
                     no positional encoding -> permutation-invariant -> B_marg.
  - tcn            : Temporal Convolutional Network (causal dilated 1-D convs).
                     ORDER-AWARE: causal convolutions read the time axis, so it
                     lies in B \\ B_marg, the order-aware part of the class.

Same hidden width (64), same minibatch Adam + grad-clip recipe, same standardization,
same leakage-controlled splits, single-threaded timing. Non-destructive: writes
results_resource_extra.json; the original results_resource_stats.json is untouched.

Usage (sanity first, then full):
    .venv/bin/python run_resource_extra.py --domains graphs --methods mil --repeats 2
    .venv/bin/python run_resource_extra.py            # all methods x 3 domains, R=20
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import time
import argparse
import resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

# Reuse the exact validated helpers (import does not run main()).
from run_resource_stats import load, split, cpu, ci95, _fit_minibatch, EPOCHS, BATCH


def _prep(sigs, y):
    """Build the standard (N, T, J) tensor frame shared by all learned baselines."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    names = sorted(sigs[0])
    J, C = len(names), len(le.classes_)
    Xseq = np.array([[s[k] for k in names] for s in sigs], dtype=np.float32).transpose(0, 2, 1)
    return Xseq, yi, J, C


def _make_model(method, J, C):
    import torch
    import torch.nn as nn

    class MIL(nn.Module):
        """Gated-attention MIL pooling (Ilse, Tomczak, Welling, ICML 2018)."""
        def __init__(self, H=64, A=64):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(J, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.V = nn.Linear(H, A)
            self.U = nn.Linear(H, A)
            self.w = nn.Linear(A, 1)
            self.head = nn.Linear(H, C)

        def forward(self, x):                       # x: (B, T, J)
            h = self.phi(x)                         # (B, T, H)
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, T, 1)
            a = torch.softmax(a, dim=1)
            z = (a * h).sum(1)                      # (B, H) order-invariant pooled
            return self.head(z)

    class SetTransformer(nn.Module):
        """Compact Set Transformer (Lee et al. 2019): a self-attention block over the
        set elements (SAB) followed by pooling-by-multihead-attention (PMA). No
        positional encoding, so it is permutation-invariant over the window."""
        def __init__(self, H=64, nh=4):
            super().__init__()
            self.proj = nn.Linear(J, H)
            self.sab = nn.MultiheadAttention(H, nh, batch_first=True)
            self.ln1 = nn.LayerNorm(H)
            self.ff = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
            self.ln2 = nn.LayerNorm(H)
            self.seed = nn.Parameter(torch.randn(1, 1, H))
            self.pma = nn.MultiheadAttention(H, nh, batch_first=True)
            self.head = nn.Linear(H, C)

        def forward(self, x):                       # x: (B, T, J)
            h = self.proj(x)
            s, _ = self.sab(h, h, h)
            h = self.ln1(h + s)
            h = self.ln2(h + self.ff(h))
            seed = self.seed.expand(h.size(0), -1, -1)
            z, _ = self.pma(seed, h, h)             # (B, 1, H) order-invariant pooled
            return self.head(z.squeeze(1))

    class TCN(nn.Module):
        """Temporal Convolutional Network: causal dilated 1-D convolutions over time.
        Order-aware (a member of B \\ B_marg)."""
        def __init__(self, H=64, dilations=(1, 2, 4)):
            super().__init__()
            self.inp = nn.Conv1d(J, H, 1)
            self.blocks = nn.ModuleList()
            for d in dilations:
                self.blocks.append(nn.ModuleDict({
                    "conv": nn.Conv1d(H, H, kernel_size=3, dilation=d, padding=0),
                    "pad": nn.ConstantPad1d((2 * d, 0), 0.0),   # left (causal) pad
                }))
            self.head = nn.Linear(H, C)

        def forward(self, x):                       # x: (B, T, J)
            h = self.inp(x.transpose(1, 2))         # (B, H, T)
            for b in self.blocks:
                y = b["conv"](b["pad"](h))
                h = torch.relu(h + y)               # residual
            return self.head(h[:, :, -1])           # last (causal) timestep

    class DeepSets(nn.Module):
        """Sum-pooling Deep Sets (Zaheer et al. 2017), order-invariant (B_marg)."""
        def __init__(self, H=64):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(J, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, C))

        def forward(self, x):
            return self.rho(self.phi(x).sum(1))

    class LSTMClf(nn.Module):
        """Order-aware recurrent baseline (LSTM, Hochreiter & Schmidhuber 1997)."""
        def __init__(self, H=64):
            super().__init__()
            self.lstm = nn.LSTM(J, H, batch_first=True)
            self.head = nn.Linear(H, C)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    return {"mil": MIL, "settransformer": SetTransformer, "tcn": TCN,
            "deepsets": DeepSets, "lstm": LSTMClf}[method]()


def run_method(method, sigs, y, blocks, repeats):
    import torch
    torch.set_num_threads(1)
    Xseq, yi, J, C = _prep(sigs, y)
    acc, tr_s, cp_s = [], [], []
    for r in range(repeats):
        torch.manual_seed(r)
        tr, te = split(y, blocks, r)
        mu = Xseq[tr].reshape(-1, J).mean(0)
        sd = Xseq[tr].reshape(-1, J).std(0) + 1e-6
        Xt = torch.tensor((Xseq - mu) / sd)
        yt = torch.tensor(yi)
        net = _make_model(method, J, C)
        c0, t0 = cpu(), time.perf_counter()
        _fit_minibatch(net, Xt[tr], yt[tr], r)
        tr_s.append(time.perf_counter() - t0)
        cp_s.append(cpu() - c0)
        net.eval()
        with torch.no_grad():
            acc.append(float((net(Xt[te]).argmax(1).numpy() == yi[te]).mean()))
    return acc, tr_s, cp_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="memory,graphs,ecology")
    ap.add_argument("--methods", default="mil,settransformer,tcn")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results_resource_extra.json")
    args = ap.parse_args()
    domains = args.domains.split(",")
    methods = args.methods.split(",")
    out = []
    for domain in domains:
        sigs, y, blocks = load(domain)
        print(f"\n### {domain} (R={args.repeats})", flush=True)
        rec = {"domain": domain}
        for m in methods:
            acc, tr_s, cp_s = run_method(m, sigs, y, blocks, args.repeats)
            rec[m] = {"acc": ci95(acc), "train_s": ci95(tr_s), "cpu_s": ci95(cp_s)}
            am, ah = rec[m]["acc"]; cm, ch = rec[m]["cpu_s"]
            print(f"  {m:14s}: acc {am:.3f}+/-{ah:.3f}  cpu {cm:.2f}+/-{ch:.2f}s", flush=True)
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, args.out), "w"), indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
