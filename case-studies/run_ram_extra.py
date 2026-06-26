"""Per-process peak RAM for the added learned baselines (cost table column), mirroring
run_resource.py: one isolated process per (method, domain). Peak RSS is reached during
training, so a short run suffices to hit the model + library footprint.

    .venv/bin/python run_ram_extra.py --method mil --domain memory
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, argparse, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)
from run_resource_stats import load
from run_resource_extra import _prep, _make_model

EP = 40


def peak():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--domain", required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1); torch.manual_seed(0)
    if args.method == "dyngraph":
        import run_resource_dyngraph as DG
        from sklearn.preprocessing import LabelEncoder
        windows, y, _ = DG.build_graph_windows()
        le = LabelEncoder().fit(y); yi = le.transform(y)
        prepped = DG.precompute(windows); net = DG.make_model(len(le.classes_))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = torch.nn.CrossEntropyLoss()
        idx = np.arange(min(32, len(yi)))
        for _ in range(EP):
            opt.zero_grad(); A, X, ws, B = DG.combine([prepped[j] for j in idx])
            lf(net.forward_batch(A, X, ws, B), torch.tensor(yi[idx])).backward(); opt.step()
    else:
        sigs, y, _ = load(args.domain)
        Xseq, yi, J, C = _prep(sigs, y)
        Xt = torch.tensor((Xseq - Xseq.reshape(-1, J).mean(0)) / (Xseq.reshape(-1, J).std(0) + 1e-6))
        yt = torch.tensor(yi); net = _make_model(args.method, J, C)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = torch.nn.CrossEntropyLoss()
        idx = np.arange(min(32, len(yi)))
        for _ in range(EP):
            opt.zero_grad(); lf(net(Xt[idx]), yt[idx]).backward(); opt.step()
    print(json.dumps({"method": args.method, "domain": args.domain,
                      "params": int(sum(p.numel() for p in net.parameters())),
                      "peak_rss_mb": peak()}))


if __name__ == "__main__":
    main()
