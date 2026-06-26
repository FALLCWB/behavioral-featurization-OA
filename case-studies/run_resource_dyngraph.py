"""Learned dynamic-graph representation baseline for the temporal-graph domain.

Reviewer feedback asked the method to be positioned against dynamic-graph
representation learning (DySAT/TGN family). Those models are domain-specific: they
consume the raw temporal graph, not the cross-domain (N,T,J) behavioral signals. So
this baseline is graphs-only and works on the SAME windows / SAME time-blocks as the
behavioral pipeline (graphs_demo.build_signals_blocked), for a head-to-head.

Model (temporal GNN, structural + temporal, plain torch, no PyTorch-Geometric):
  per sub-bin -> 2-layer GCN over the active-node graph (normalized adjacency
  propagation, node features = [1, deg, log1p(deg)]) -> mean-pool nodes to a sub-bin
  graph embedding -> GRU over the K sub-bins -> window embedding -> linear classifier.
The active node set is unknown a priori and volatile across sub-bins (exactly the
trigger condition); the GNN handles variable node counts natively and graph-pooling
yields a fixed window embedding.

Same fair protocol as the other learned baselines: hidden width comparable, minibatch
Adam + grad clip, single-threaded timing, blocked leakage-controlled CV, R repeats.

Usage:
    .venv/bin/python run_resource_dyngraph.py --repeats 2      # sanity
    .venv/bin/python run_resource_dyngraph.py --repeats 20     # full
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
for d in ["common", "graphs-dynamic"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

from run_resource_stats import cpu, ci95, EPOCHS, BATCH
import graphs_demo as G


def build_graph_windows(n_blocks=5):
    """Same windows/blocks as graphs_demo.build_signals_blocked, but emitting the raw
    sub-bin graph structure (per window: a list of K relabeled edge arrays + node
    counts) instead of the behavioral signals."""
    here = os.path.dirname(G.__file__)
    windows, y, block = [], [], []
    for name, fn in G.NETS.items():
        edges = G.load_edges(os.path.join(here, "data", fn))
        nwin = len(edges) // G.WIN
        for i in range(nwin):
            we = edges[i * G.WIN:(i + 1) * G.WIN]
            per = max(1, len(we) // G.K)
            bins = [we[j * per:(j + 1) * per] for j in range(G.K)]
            bin_edge_sets = []
            subgraphs = []
            for b in bins:
                es = set((min(u, v), max(u, v)) for u, v, _ in b)
                bin_edge_sets.append(es)
                recent = bin_edge_sets[-G.HORIZON:]
                active = set().union(*recent) if recent else set()
                subgraphs.append(active)
            windows.append(subgraphs)
            y.append(name)
            block.append(min(n_blocks - 1, i * n_blocks // nwin))
    return windows, np.array(y), np.array(block)


def _norm_adj_and_feats(active_edges):
    """Build normalized adjacency (with self-loops) and structural node features for
    one sub-bin graph. Returns (A_hat [m,m], X [m,3]); empty graph -> single dummy."""
    nodes = sorted({u for e in active_edges for u in e})
    if not nodes:
        return np.eye(1, dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
    idx = {u: k for k, u in enumerate(nodes)}
    m = len(nodes)
    A = np.eye(m, dtype=np.float32)
    for (u, v) in active_edges:
        a, b = idx[u], idx[v]
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(deg)
    A_hat = (A * dinv[:, None]) * dinv[None, :]
    raw_deg = deg - 1.0  # subtract self-loop
    X = np.stack([np.ones(m), raw_deg, np.log1p(raw_deg)], axis=1).astype(np.float32)
    return A_hat, X


def precompute(windows):
    """Per window, build ONE SPARSE block-diagonal graph stacking the K sub-bins (same
    GCN math, single sparse forward instead of K dense ones), plus a segment index
    mapping each node to its sub-bin so a segment-mean recovers the (K, H) per-sub-bin
    embeddings. The sub-bin graphs are sparse (~10^4 nonzeros/window vs ~10^7 dense),
    so a sparse block-diagonal is the correct efficient implementation to time."""
    import torch
    prepped = []
    for subgraphs in windows:
        rows, cols, vals, feats, seg = [], [], [], [], []
        off = 0
        for k, active in enumerate(subgraphs):
            A, X = _norm_adj_and_feats(active)
            nz = np.nonzero(A)
            rows.append(nz[0] + off); cols.append(nz[1] + off); vals.append(A[nz])
            feats.append(X); seg.extend([k] * X.shape[0]); off += X.shape[0]
        idx = np.stack([np.concatenate(rows), np.concatenate(cols)])
        Asp = torch.sparse_coo_tensor(torch.tensor(idx, dtype=torch.long),
                                      torch.tensor(np.concatenate(vals), dtype=torch.float32),
                                      size=(off, off)).coalesce()
        prepped.append((Asp, torch.tensor(np.concatenate(feats, 0)),
                        torch.tensor(np.asarray(seg), dtype=torch.long)))
    return prepped


def make_model(C, H=32, K=None):
    import torch
    import torch.nn as nn
    KK = K if K is not None else G.K

    class TemporalGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(3, H)
            self.w2 = nn.Linear(H, H)
            self.gru = nn.GRU(H, H, batch_first=True)
            self.head = nn.Linear(H, C)

        def forward_batch(self, A, X, win_seg, B):  # one combined sparse graph for B windows
            h = torch.relu(torch.sparse.mm(A, self.w1(X)))
            h = torch.relu(torch.sparse.mm(A, self.w2(h)))   # (Ntot_all, H)
            HH = h.shape[1]
            emb = torch.zeros(B * KK, HH).index_add_(0, win_seg, h)
            cnt = torch.zeros(B * KK).index_add_(0, win_seg, torch.ones(h.shape[0]))
            emb = (emb / cnt.clamp(min=1.0).unsqueeze(1)).view(B, KK, HH)
            _, hk = self.gru(emb)
            return self.head(hk[-1])               # (B, C)

    return TemporalGNN()


def combine(items, K=None):
    """Block-diagonal-combine B per-window (Asp, X, seg) into ONE sparse graph + a
    per-window segment index, so a minibatch is a single GCN forward."""
    import torch
    KK = K if K is not None else G.K
    idxs, vals, feats, wseg, off = [], [], [], [], 0
    for b, (A, X, seg) in enumerate(items):
        ix = A.indices() + off
        idxs.append(ix); vals.append(A.values()); feats.append(X)
        wseg.append(b * KK + seg)
        off += X.shape[0]
    Abig = torch.sparse_coo_tensor(torch.cat(idxs, 1), torch.cat(vals),
                                   size=(off, off)).coalesce()
    return Abig, torch.cat(feats, 0), torch.cat(wseg), len(items)


def split(y, blocks, r):
    b = sorted(set(blocks.tolist()))[r % len(set(blocks.tolist()))]
    te = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]
    return tr, te


def run(repeats):
    import torch
    from sklearn.preprocessing import LabelEncoder
    torch.set_num_threads(1)
    windows, y, blocks = build_graph_windows()
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    C = len(le.classes_)
    prepped = precompute(windows)
    acc, cp_s, tr_s = [], [], []
    for r in range(repeats):
        torch.manual_seed(r)
        tr, te = split(y, blocks, r)
        net = make_model(C)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(r)
        c0, t0 = cpu(), time.perf_counter()
        net.train()
        for _ in range(EPOCHS):
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                A, X, ws, B = combine([prepped[j] for j in idx])
                loss = lossf(net.forward_batch(A, X, ws, B), torch.tensor(yi[idx]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        tr_s.append(time.perf_counter() - t0)
        cp_s.append(cpu() - c0)
        net.eval()
        with torch.no_grad():
            A, X, ws, B = combine([prepped[j] for j in te])
            pred = net.forward_batch(A, X, ws, B).argmax(1).numpy()
        acc.append(float((pred == yi[te]).mean()))
        print(f"  rep {r}: acc {acc[-1]:.3f}  cpu {cp_s[-1]:.2f}s", flush=True)
    return acc, tr_s, cp_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results_resource_dyngraph.json")
    args = ap.parse_args()
    print(f"### graphs dynamic-GNN (R={args.repeats})", flush=True)
    acc, tr_s, cp_s = run(args.repeats)
    rec = {"domain": "graphs", "method": "dyngraph_gnn",
           "acc": ci95(acc), "train_s": ci95(tr_s), "cpu_s": ci95(cp_s)}
    print(f"  dyngraph_gnn: acc {rec['acc'][0]:.3f}+/-{rec['acc'][1]:.3f}  "
          f"cpu {rec['cpu_s'][0]:.2f}+/-{rec['cpu_s'][1]:.2f}s", flush=True)
    json.dump(rec, open(os.path.join(HERE, args.out), "w"), indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
