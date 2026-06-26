"""Synthetic generativity experiment for Behavioral Featurization.

Canonical permutation-invariance construction: each correlated (Markov) sequence is
paired with a SHUFFLE of itself. Shuffling preserves the multiset of states exactly
and destroys temporal order. A permutation-invariant set-pool or a stream sketch sees
an identical input for a sequence and its shuffle, so it CANNOT tell the two classes
apart (chance by construction). Transition-aware behavioral features separate them
trivially. This is the executable confirmation of the Proposition.
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

P_STICKY = np.array([[0.9, 0.1], [0.1, 0.9]])   # correlated source (rarely switches)
LEN = 200


def sample_chain(P, n, rng):
    s = int(rng.integers(0, 2))
    seq = [s]
    for _ in range(n - 1):
        s = int(rng.choice(2, p=P[s]))
        seq.append(s)
    return np.array(seq)


def marginal_feature(seq):
    # multiset summary: exactly what a permutation-invariant set-pool / sketch retains.
    # Invariant to shuffling by construction.
    return np.array([np.mean(seq == 0), np.mean(seq == 1)])


def behavioral_feature(seq):
    # transition-aware: switching rate and lag-1 autocorrelation
    switch_rate = np.mean(seq[1:] != seq[:-1])
    x = seq.astype(float)
    m, s = x.mean(), x.std()
    ac1 = float(np.mean((x[:-1] - m) * (x[1:] - m)) / (s * s)) if s > 0 else 0.0
    return np.array([switch_rate, ac1])


def build(n_per=200, seed=0):
    """Independent draws: 'ordered' = correlated sticky sequences; 'shuffled' = other
    sticky sequences with their order destroyed. Shuffling preserves per-sequence
    counts, so both classes share an identical marginal (multiset) distribution while
    differing only in temporal structure. Independent draws (no exact twins) avoid an
    identical-input/opposite-label collision in cross-validation."""
    rng = np.random.default_rng(seed)
    seqs, y = [], []
    for _ in range(n_per):
        seqs.append(sample_chain(P_STICKY, LEN, rng))
        y.append("ordered")
    for _ in range(n_per):
        s = sample_chain(P_STICKY, LEN, rng)
        rng.shuffle(s)
        seqs.append(s)
        y.append("shuffled")
    return seqs, np.array(y)


def main():
    here = os.path.dirname(__file__)
    figdir = os.path.join(here, "..", "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    seqs, y = build()
    Xm = np.array([marginal_feature(s) for s in seqs])
    Xb = np.array([behavioral_feature(s) for s in seqs])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def rf():
        return RandomForestClassifier(n_estimators=200, random_state=0)

    acc_m = float(cross_val_score(rf(), Xm, y, cv=skf).mean())
    acc_b = float(cross_val_score(rf(), Xb, y, cv=skf).mean())

    print("=== Generativity (Proposition) - shuffle / permutation-invariance experiment ===")
    print(f"state-0 frequency:  ordered={Xm[y=='ordered',0].mean():.3f}  "
          f"shuffled={Xm[y=='shuffled',0].mean():.3f}   (matched marginal distribution)")
    print(f"accuracy, MARGINAL-only (set-pool / sketch) features: {acc_m:.3f}   (~ chance 0.5)")
    print(f"accuracy, BEHAVIORAL (transition) features:           {acc_b:.3f}")

    jit = np.random.default_rng(1).normal(0, 1, len(y))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"ordered": "tab:purple", "shuffled": "tab:orange"}
    for lab in ["ordered", "shuffled"]:
        m = y == lab
        ax[0].scatter(Xm[m, 0], jit[m], s=12, color=colors[lab], alpha=0.5, label=lab)
        ax[1].scatter(Xb[m, 0], Xb[m, 1], s=12, color=colors[lab], alpha=0.5, label=lab)
    ax[0].set_xlabel("marginal feature (state-0 frequency)")
    ax[0].set_yticks([])
    ax[0].set_title(f"marginal-only view (accuracy {acc_m:.2f})")
    ax[0].legend()
    ax[1].set_xlabel("switching rate")
    ax[1].set_ylabel("lag-1 autocorrelation")
    ax[1].set_title(f"behavioral view (accuracy {acc_b:.2f})")
    ax[1].legend()
    fig.tight_layout()
    fp = os.path.join(figdir, "F3_generativity.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print("figure saved:", os.path.relpath(fp, here))
    with open(os.path.join(here, "results_generativity.json"), "w") as f:
        json.dump({"acc_marginal": acc_m, "acc_behavioral": acc_b}, f, indent=2)


if __name__ == "__main__":
    main()
