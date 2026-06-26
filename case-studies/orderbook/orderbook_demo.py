"""Limit order book case study: the CONJUNCTION case.

The evolving object is a real limit order book (Binance BTCUSDT depth snapshots).
Its dimensionality is VOLATILE: the number of occupied price levels and the price
grid shift over time. The natural label is ORDER-DEPENDENT: the direction of the
future mid-price move, which depends on the temporal dynamics of order-flow
imbalance and price, not on a static snapshot. This is the case where both axes of
the trigger condition hold at once: volatile dimensionality AND a label for which
temporal order is decisive.

Per snapshot the variable-size book is reduced to stationary scalar signals
(returns, order-flow imbalance, spread, depth, number of active levels); a window of
these is the evolving object, featurized by the shared compact Psi into a fixed
vector. Windows are non-overlapping with time-block group ids for leakage-controlled
blocked CV.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "BTCUSDT_lob.npz")

WIN = 40            # snapshots per window (the evolving object, ~14 s)
HORIZON = 240       # predict mid direction ~86 s ahead (long enough that the mid is
                    # rarely unchanged; at short horizons it is sticky -> ties at zero)
N_BLOCKS = 5        # contiguous time blocks for grouped CV
# Label is balanced by a MEDIAN split of the future move: predicting whether the next
# move is above or below the typical one removes the session drift (which would make
# raw sign trivially imbalanced) and isolates the order-dependent relative direction.


def _snapshot_features(bp, bs, ap, asz):
    """Reduce one variable-size book snapshot to stationary scalars."""
    best_bid, best_ask = bp[0], ap[0]
    mid = 0.5 * (best_bid + best_ask)
    bid_depth = bs.sum(); ask_depth = asz.sum()
    tot = bid_depth + ask_depth
    imb = (bid_depth - ask_depth) / tot if tot > 0 else 0.0
    n_levels = float((bs > 0).sum() + (asz > 0).sum())   # volatile dimensionality
    denom = bs[0] + asz[0]
    micro = (best_bid * asz[0] + best_ask * bs[0]) / denom if denom > 0 else mid
    spread = (best_ask - best_bid) / mid if mid > 0 else 0.0
    return mid, imb, spread, np.log1p(bid_depth), np.log1p(ask_depth), n_levels, (micro - mid) / mid


def load_lob():
    d = np.load(NPZ)
    return d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"]


def _series():
    bp, bs, ap, asz = load_lob()
    n = bp.shape[0]
    mids = np.empty(n); imb = np.empty(n); spr = np.empty(n)
    bd = np.empty(n); ad = np.empty(n); nl = np.empty(n); mj = np.empty(n)
    for t in range(n):
        mids[t], imb[t], spr[t], bd[t], ad[t], nl[t], mj[t] = _snapshot_features(
            bp[t], bs[t], ap[t], asz[t])
    return mids, dict(imbalance=imb, spread=spr, bid_depth=bd, ask_depth=ad,
                      nlevels=nl, micro_adj=mj)


def build_signals_blocked():
    """Non-overlapping windows; label = sign of mid move HORIZON snapshots after the
    window end; middle deadzone dropped. Returns (sigs, y, blocks)."""
    mids, feats = _series()
    logmid = np.log(mids)
    n = len(mids)
    starts = list(range(0, n - WIN - HORIZON, WIN))   # non-overlapping
    moves = np.array([logmid[s + WIN + HORIZON] - logmid[s + WIN] for s in starts])
    thr = np.median(moves)                            # balanced median split (drift-removed)
    nstart = len(starts)
    sigs, y, blocks = [], [], []
    for j, s in enumerate(starts):
        e = s + WIN
        sl = slice(s, e)
        ret = np.diff(logmid[s:e + 1])               # WIN returns (order-sensitive)
        sig = {"ret": list(ret)}
        for k, v in feats.items():
            sig[k] = list(v[sl])
        sigs.append(sig)
        y.append("up" if moves[j] > thr else "down")
        blocks.append(min(N_BLOCKS - 1, j * N_BLOCKS // max(nstart, 1)))
    return sigs, np.array(y), np.array(blocks)


def info():
    d = np.load(NPZ)
    n = d["bid_p"].shape[0]
    span = (d["ts"][-1] - d["ts"][0]) / 60.0
    return n, span
