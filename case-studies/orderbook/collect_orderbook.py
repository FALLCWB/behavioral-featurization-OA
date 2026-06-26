"""Collect a real limit-order-book time series from the public Binance REST depth
endpoint. Each snapshot is the live order book (top-K levels per side); polled at a
fixed cadence so the sequence of snapshots is a real evolving object whose temporal
ORDER carries the short-horizon price signal. Saved compactly as .npz.

No API key required; respects the weight budget (limit=100 -> weight 5).
"""
from __future__ import annotations
import os
import sys
import time
import json
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOL = "BTCUSDT"
K = 100                 # levels per side (Binance limit=100, weight 5)
URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit={K}"


def fetch():
    with urllib.request.urlopen(URL, timeout=10) as r:
        d = json.loads(r.read().decode())
    b = np.array(d["bids"], dtype=float)   # (K,2) price,size  (desc)
    a = np.array(d["asks"], dtype=float)   # (K,2) price,size  (asc)
    return b, a


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    bid_p = np.zeros((n_target, K), np.float64); bid_s = np.zeros((n_target, K), np.float64)
    ask_p = np.zeros((n_target, K), np.float64); ask_s = np.zeros((n_target, K), np.float64)
    ts = np.zeros(n_target, np.float64)
    got = 0
    out = os.path.join(HERE, "data", f"{SYMBOL}_lob.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    while got < n_target:
        t0 = time.time()
        try:
            b, a = fetch()
            if b.shape[0] >= 1 and a.shape[0] >= 1:
                nb, na = min(K, b.shape[0]), min(K, a.shape[0])
                bid_p[got, :nb] = b[:nb, 0]; bid_s[got, :nb] = b[:nb, 1]
                ask_p[got, :na] = a[:na, 0]; ask_s[got, :na] = a[:na, 1]
                ts[got] = t0
                got += 1
                if got % 500 == 0:
                    np.savez_compressed(out, ts=ts[:got], bid_p=bid_p[:got], bid_s=bid_s[:got],
                                        ask_p=ask_p[:got], ask_s=ask_s[:got], K=K, symbol=SYMBOL)
                    print(f"  {got}/{n_target} snapshots saved", flush=True)
        except Exception as e:
            print(f"  transient: {type(e).__name__}", flush=True)
            time.sleep(1.0)
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)
    np.savez_compressed(out, ts=ts[:got], bid_p=bid_p[:got], bid_s=bid_s[:got],
                        ask_p=ask_p[:got], ask_s=ask_s[:got], K=K, symbol=SYMBOL)
    span = (ts[got - 1] - ts[0]) / 60.0
    print(f"DONE: {got} snapshots over {span:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
