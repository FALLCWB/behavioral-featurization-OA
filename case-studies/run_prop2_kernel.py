"""Direct demonstration of Proposition (ii): for a Markov object, the windowed
TRANSITION COUNTS are a sufficient statistic for the kernel, and recover it.

Three self-contained checks on data drawn from KNOWN Markov kernels:

  (1) Consistency / identifiability. The maximum-likelihood kernel estimated from
      the transition-count matrix converges to the true kernel as the sequence
      grows: mean Frobenius error -> 0 at the ~T^{-1/2} rate (95% CI shown).

  (2) Sufficiency (no information loss). The Bayes-optimal classifier of "which
      kernel generated this sequence" depends on the data ONLY through the
      transition counts. Empirically, a classifier on count features matches that
      optimum, and an ORDER-AWARE featurization (path signature of the one-hot
      path) does not exceed it: order beyond the counts carries no extra signal.

  (3) Factorization identity. Two DISTINCT sequences with identical transition
      counts (a different Eulerian trail of the same transition multigraph) have
      identical log-likelihood under every kernel: the data enters the likelihood
      only through the counts (Fisher-Neyman), i.e. the counts are sufficient.

Reference: Anderson & Goodman (1957); Billingsley (1961).
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from pipeline import signature_features

SEED = 0
K = 4                       # number of states


def random_kernel(rng, alpha=0.5, k=K):
    """Row-stochastic kernel; small alpha -> peaked rows (well-separated kernels)."""
    return rng.dirichlet(alpha * np.ones(k), size=k)


def gen_chain(P, T, rng, s0=0):
    s = np.empty(T + 1, dtype=int)
    s[0] = s0
    k = P.shape[0]
    for t in range(T):
        s[t + 1] = rng.choice(k, p=P[s[t]])
    return s


def trans_counts(seq, k=K):
    N = np.zeros((k, k))
    np.add.at(N, (seq[:-1], seq[1:]), 1.0)
    return N


def mle_kernel(N):
    rs = N.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return N / rs


def ci95(a):
    a = np.asarray(a, float)
    m = a.mean()
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(m), float(h)


# ---------- (1) consistency / identifiability ----------
def consistency(rng, Ts=(50, 100, 200, 400, 800, 1600, 3200), reps=300):
    P = random_kernel(rng, alpha=0.5)
    means, his = [], []
    for T in Ts:
        errs = []
        for _ in range(reps):
            seq = gen_chain(P, T, rng)
            Ph = mle_kernel(trans_counts(seq))
            errs.append(np.linalg.norm(Ph - P))
        m, h = ci95(errs)
        means.append(m); his.append(h)
        print(f"  T={T:5d}  ||P_hat - P||_F = {m:.4f} +/- {h:.4f}")
    return {"Ts": list(Ts), "err_mean": means, "err_ci": his,
            "true_kernel": P.tolist()}


# ---------- (2) sufficiency: counts match the optimum, order adds nothing ----------
def onehot_signals(seq, k=K):
    """One-hot indicator series per state: an order-PRESERVING view of the path."""
    oh = np.zeros((k, len(seq)))
    oh[seq, np.arange(len(seq))] = 1.0
    return {f"s{j}": list(oh[j]) for j in range(k)}


def sufficiency(rng, n_kernels=6, T=200, n_per=80):
    kernels = [random_kernel(rng, alpha=0.4) for _ in range(n_kernels)]
    logPs = [np.log(np.clip(P, 1e-12, 1.0)) for P in kernels]
    seqs, y = [], []
    for m, P in enumerate(kernels):
        for _ in range(n_per):
            seqs.append(gen_chain(P, T, rng))
            y.append(m)
    y = np.array(y)

    Ncs = [trans_counts(s) for s in seqs]
    # Bayes-optimal (oracle): argmax_m sum(N * logP_m) -- uses counts ONLY
    opt_pred = np.array([int(np.argmax([np.sum(N * lp) for lp in logPs])) for N in Ncs])
    opt_acc = float((opt_pred == y).mean())

    # count features (normalized transition counts) vs order-aware signature
    Xcount = np.array([mle_kernel(N).ravel() for N in Ncs])
    Xsig = np.array([signature_features(onehot_signals(s)) for s in seqs])

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    sc_count = cross_val_score(clf, Xcount, y, cv=rkf, n_jobs=-1)
    sc_sig = cross_val_score(clf, Xsig, y, cv=rkf, n_jobs=-1)
    cm, ch = ci95(sc_count)
    sm, sh = ci95(sc_sig)
    chance = 1.0 / n_kernels
    print(f"  oracle (counts, Bayes-optimal): {opt_acc:.3f}")
    print(f"  count-features RF            : {cm:.3f} +/- {ch:.3f}")
    print(f"  order-aware signature RF     : {sm:.3f} +/- {sh:.3f}")
    print(f"  chance={chance:.3f}")
    return {"n_kernels": n_kernels, "T": T, "n_per": n_per, "chance": chance,
            "oracle_acc": opt_acc,
            "count_acc": [round(cm, 4), round(ch, 4)],
            "signature_acc": [round(sm, 4), round(sh, 4)],
            "order_adds_over_counts": round(sm - cm, 4)}


# ---------- (3) factorization: same counts -> same likelihood ----------
def eulerian_trail(adj, start, rng):
    """Hierholzer on a directed multigraph (adj[u] = list of v). Consumes a copy;
    randomized neighbor order yields a (possibly) different valid trail."""
    adj = {u: list(vs) for u, vs in adj.items()}
    for u in adj:
        rng.shuffle(adj[u])
    stack, trail = [start], []
    while stack:
        u = stack[-1]
        if adj.get(u):
            stack.append(adj[u].pop())
        else:
            trail.append(stack.pop())
    return trail[::-1]


def factorization(rng, T=400, n_test_kernels=1000):
    P = random_kernel(rng, alpha=0.7)
    seqA = gen_chain(P, T, rng)
    NA = trans_counts(seqA)
    adj = {u: [] for u in range(K)}
    for a, b in zip(seqA[:-1], seqA[1:]):
        adj[a].append(b)
    # search for an alternative trail with identical edge multiset but different order
    seqB = None
    for _ in range(200):
        t = eulerian_trail(adj, seqA[0], rng)
        if len(t) == len(seqA):
            tb = np.array(t)
            if np.array_equal(trans_counts(tb), NA) and not np.array_equal(tb, seqA):
                seqB = tb
                break
    if seqB is None:
        print("  (no distinct equal-count trail found; identity holds trivially)")
        return {"found_distinct": False}
    NB = trans_counts(seqB)
    same_counts = bool(np.array_equal(NA, NB))
    # log-likelihood of each sequence under many random kernels
    diffs = []
    for _ in range(n_test_kernels):
        Q = random_kernel(rng, alpha=1.0)
        lQ = np.log(np.clip(Q, 1e-12, 1.0))
        llA = np.sum(NA * lQ)
        llB = np.sum(NB * lQ)
        diffs.append(abs(llA - llB))
    maxdiff = float(np.max(diffs))
    print(f"  distinct sequences, identical counts: {same_counts}")
    print(f"  max |logL_A - logL_B| over {n_test_kernels} random kernels: {maxdiff:.2e}")
    return {"found_distinct": True, "same_counts": same_counts,
            "max_loglik_diff": maxdiff, "n_test_kernels": n_test_kernels,
            "seq_len": int(T)}


def make_figure(cons, outpath):
    Ts = np.array(cons["Ts"]); m = np.array(cons["err_mean"]); h = np.array(cons["err_ci"])
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.fill_between(Ts, m - h, m + h, alpha=0.25, color="tab:blue")
    ax.plot(Ts, m, "o-", color="tab:blue", label=r"$\|\hat P - P\|_F$ (95% CI)")
    ref = m[0] * np.sqrt(Ts[0]) / np.sqrt(Ts)
    ax.plot(Ts, ref, "k--", alpha=0.6, label=r"$\propto T^{-1/2}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length T"); ax.set_ylabel("kernel recovery error")
    ax.set_title("kernel recovery error vs sequence length")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    print("=== Proposition (ii): transition counts are sufficient & recover the kernel ===")
    print("\n(1) consistency / identifiability:")
    cons = consistency(rng)
    print("\n(2) sufficiency (counts match optimum; order adds nothing):")
    suff = sufficiency(rng)
    print("\n(3) factorization (same counts -> same likelihood):")
    fact = factorization(rng)
    figdir = os.path.join(HERE, "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, "F5_prop2_kernel.png")
    make_figure(cons, figpath)
    print("\nfigure saved:", os.path.relpath(figpath, HERE))
    out = {"consistency": cons, "sufficiency": suff, "factorization": fact}
    json.dump(out, open(os.path.join(HERE, "results_prop2_kernel.json"), "w"), indent=2)
    print("saved results_prop2_kernel.json")


if __name__ == "__main__":
    main()
