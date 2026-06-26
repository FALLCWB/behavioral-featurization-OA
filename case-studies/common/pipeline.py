"""Shared 'Behavioral Featurization' pipeline.

The uniform recipe, applied identically across every domain:
    evolving object O_t -> behavioral signals b^(j)(t) -> Psi (fixed featurizer)
    -> z (fixed vector) -> classifier g.

This single module is reused by all case studies; reusing the same code IS the
generality argument in executable form. Psi here is a compact, documented feature
set spanning the catch22 design families (distribution, trend, autocorrelation,
successive differences). catch22 / tsfresh are drop-in replacements for Psi in the
paper's final runs.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURE_NAMES = [
    "mean", "std", "min", "max", "range", "median", "iqr", "slope",
    "ac1", "ac2", "mean_abs_diff", "std_diff", "mean_crossings",
    "prop_above_mean", "value_entropy", "net_change",
]


def featurize_signal(x) -> np.ndarray:
    """Psi: map a 1-D behavioral signal of any length to a fixed feature vector."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros(len(FEATURE_NAMES))
    mean = x.mean()
    std = x.std()
    mn, mx = x.min(), x.max()
    median = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    t = np.arange(n)
    slope = np.polyfit(t, x, 1)[0] if (n > 1 and np.ptp(t) > 0) else 0.0

    def autocorr(lag):
        if n <= lag or std == 0:
            return 0.0
        a = x[:-lag] - mean
        b = x[lag:] - mean
        denom = (n - lag) * std * std
        return float(np.sum(a * b) / denom) if denom != 0 else 0.0

    d = np.diff(x) if n > 1 else np.array([0.0])
    mean_crossings = (np.sum((x[:-1] - mean) * (x[1:] - mean) < 0) / max(n - 1, 1)) if n > 1 else 0.0
    hist, _ = np.histogram(x, bins=min(10, max(2, n // 2)))
    p = hist[hist > 0] / hist.sum()
    value_entropy = float(-(p * np.log(p)).sum()) if p.size else 0.0

    return np.array([
        mean, std, mn, mx, mx - mn, median, iqr, slope,
        autocorr(1), autocorr(2), np.mean(np.abs(d)), np.std(d),
        mean_crossings, np.mean(x > mean), value_entropy, x[-1] - x[0],
    ], dtype=float)


def featurize_signals(signals: dict) -> np.ndarray:
    """Concatenate Psi over each named behavioral signal -> fixed vector z."""
    return np.concatenate([featurize_signal(signals[name]) for name in sorted(signals)])


def feature_vector_names(signal_names) -> list:
    return [f"{s}::{f}" for s in sorted(signal_names) for f in FEATURE_NAMES]


def pool_features(signals) -> np.ndarray:
    """Permutation-invariant pooled statistics per signal (Deep Sets-style baseline:
    depends only on the multiset of values, discards temporal order)."""
    parts = []
    for name in sorted(signals):
        x = np.asarray(signals[name], dtype=float)
        parts.append(np.array([x.mean(), x.std(), x.min(), x.max(), x.sum()]) if x.size else np.zeros(5))
    return np.concatenate(parts)


def sketch_features(signals) -> np.ndarray:
    """Fixed-size quantile sketch per signal (synopsis baseline: a histogram summary
    of the multiset of values, order-invariant)."""
    qs = list(range(0, 101, 10))
    parts = []
    for name in sorted(signals):
        x = np.asarray(signals[name], dtype=float)
        parts.append(np.percentile(x, qs) if x.size else np.zeros(len(qs)))
    return np.concatenate(parts)


def naive_features(signals) -> np.ndarray:
    """Naive single-snapshot baseline: only the final value of each signal."""
    return np.array([float(np.asarray(signals[name], dtype=float)[-1]) if len(signals[name]) else 0.0
                     for name in sorted(signals)])


def signature_features(signals, time_aug=True) -> np.ndarray:
    """Truncated depth-2 path signature (left-Riemann), time-augmented. This is an
    ORDER-AWARE fixed-length featurization of a (multivariate) path, the canonical
    order-preserving representation of variable-length streams (Lyons; Chevyrev &
    Kormilitzin 2016). Used as a strong baseline that, unlike pooling/sketching,
    retains temporal order. Each signal is z-normalized; level 1 is the net change,
    level 2 (the iterated integrals, including signed area) encodes order."""
    names = sorted(signals)
    cols = []
    for n in names:
        x = np.asarray(signals[n], dtype=float)
        s = x.std()
        cols.append((x - x.mean()) / s if s > 0 else x * 0.0)
    P = np.array(cols).T
    if time_aug:
        T = P.shape[0]
        P = np.hstack([np.linspace(0, 1, T).reshape(-1, 1), P])
    d = P.shape[1]
    if P.shape[0] < 2:
        return np.zeros(d + d * d)
    dP = np.diff(P, axis=0)
    base = P[:-1] - P[0]
    level1 = P[-1] - P[0]
    level2 = base.T @ dP
    return np.concatenate([level1, level2.ravel()])


def shuffle_signals(signals, rng):
    """Permute the time index identically across all signals: preserves the multiset
    of (joint) states, destroys temporal order. For the ordered-vs-shuffled diagnostic."""
    names = list(signals)
    T = len(signals[names[0]])
    perm = rng.permutation(T)
    return {n: list(np.asarray(signals[n], dtype=float)[perm]) for n in names}


def tsfresh_featurize_batch(sigs, kind="efficient"):
    """Featurize a list of signal-dicts with tsfresh (an established featurizer).
    Runs tsfresh per signal across all windows, then concatenates. Returns (X, names)."""
    import pandas as pd
    from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
    params = MinimalFCParameters() if kind == "minimal" else EfficientFCParameters()
    signal_names = sorted(sigs[0])
    blocks = []
    for sname in signal_names:
        rows = []
        for wid, s in enumerate(sigs):
            for t, v in enumerate(s[sname]):
                rows.append((wid, t, float(v)))
        df = pd.DataFrame(rows, columns=["id", "time", "val"])
        f = extract_features(df, column_id="id", column_sort="time", column_value="val",
                             default_fc_parameters=params, disable_progressbar=True, n_jobs=0)
        f = f.reindex(range(len(sigs)))  # keep window order
        f.columns = [f"{sname}__{c}" for c in f.columns]
        blocks.append(f.reset_index(drop=True))
    X = pd.concat(blocks, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.values, list(X.columns)


def run_classification(X, y, n_splits=5, seed=0, model=None) -> dict:
    """Train classifier g with stratified CV; return accuracy, CV spread, confusion."""
    X = np.asarray(X)
    y = np.asarray(y)
    if model is None:
        model = RandomForestClassifier(n_estimators=300, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "labels": sorted(set(y.tolist())),
        "confusion": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, digits=3),
    }
