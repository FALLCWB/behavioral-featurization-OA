"""Is the graph forecasting series mean-reverting, and does the level beat the trend?

Section V-E accounts for the dynamic-graph forecasting result, where the
marginal-only pool reaches a higher accuracy than the order-retaining
representation, by stating that the series are mean-reverting and that the current
level rather than its trend predicts the next move. This measures both clauses on
exactly the series the forecasting label is built from: the per-window mean of each
source's primary signal, the quantity whose direction of change is the label.

What is computed:

  lag-1 reversion   Pearson correlation between a window's deviation from its
                    source's own mean and the next change, with a 95% interval
                    from a cluster bootstrap over the same five contiguous time
                    blocks the leakage-controlled folds use, which is the
                    conservative unit on a blocked domain. Negative is mean
                    reversion: above average now, down next.
  level rule        accuracy of the zero-parameter rule "above the source mean,
                    predict down; below, predict up".
  trend rule        accuracy of the momentum rule "it went up last step, predict
                    up", the order-dependent alternative.

Both rules are reported twice. The descriptive figures use every window and a
threshold read from the whole series. The figures the paper quotes are evaluated
under the same StratifiedGroupKFold over the five time blocks that the forecasting
experiment uses, with each source's threshold estimated on the training blocks
alone, so that no held-out block informs the threshold and the accuracies can be
set beside the leakage-controlled accuracy of the learned pool.

Usage: python3 run_mean_reversion.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from scipy import stats as sps
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))

import run_forecasting as FC   # the same window construction as the label

SEED = 0
BOOT = 2000
N_BLOCKS = FC.N_BLOCKS


def series(sources, signal_key):
    """Per-source window means, the series whose next change is the label."""
    out = {}
    for src, wins in sources.items():
        out[src] = np.array([float(np.mean(w[signal_key])) for w in wins])
    return out


def points(sources, signal_key):
    """(deviation, next change, previous change, label, block, level, source) per point."""
    dev, nxt, prev, lab, blk, lvl, who = [], [], [], [], [], [], []
    for si, (src, x) in enumerate(series(sources, signal_key).items()):
        m = len(x) - 1
        mu = float(x.mean())
        for i in range(m):
            dev.append(x[i] - mu)
            nxt.append(x[i + 1] - x[i])
            prev.append(x[i] - x[i - 1] if i > 0 else 0.0)
            lab.append(1 if x[i + 1] > x[i] else 0)
            blk.append(min(N_BLOCKS - 1, i * N_BLOCKS // max(m, 1)))
            lvl.append(x[i])
            who.append(si)
    return (np.array(dev), np.array(nxt), np.array(prev),
            np.array(lab), np.array(blk), np.array(lvl), np.array(who))


def stats(dev, nxt, prev, lab):
    r = float(np.corrcoef(dev, nxt)[0, 1]) if dev.std() > 0 and nxt.std() > 0 else float("nan")
    level = float(np.mean((dev < 0).astype(int) == lab))     # below mean -> predict up
    trend = float(np.mean((prev > 0).astype(int) == lab))    # went up -> predict up
    return r, level, trend


def boot_ci(dev, nxt, prev, lab, blk):
    rng = np.random.default_rng(SEED)
    blocks = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in blocks}
    keep = {k: [] for k in ("r", "level", "trend")}
    for _ in range(BOOT):
        take = np.concatenate([idx[b] for b in rng.choice(blocks, len(blocks), replace=True)])
        r, lv, tr = stats(dev[take], nxt[take], prev[take], lab[take])
        if not np.isnan(r):
            keep["r"].append(r)
        keep["level"].append(lv)
        keep["trend"].append(tr)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in keep.items()}


def blocked_rules(lvl, prev, lab, blk, who):
    """The two rules and the majority baseline under the leakage-controlled folds.

    The level rule needs a threshold, and a threshold read from the whole series
    would look ahead. Here each source's mean is estimated on the training blocks
    of the fold and applied to the held-out block, which is the protocol the
    forecasting experiment itself uses.
    """
    cv = StratifiedGroupKFold(5)
    acc = {"level": [], "trend": [], "majority": []}
    for tr, te in cv.split(lvl.reshape(-1, 1), lab, groups=blk):
        mu = {s: lvl[tr][who[tr] == s].mean() for s in np.unique(who[tr])}
        thr = np.array([mu.get(s, lvl[tr].mean()) for s in who[te]])
        acc["level"].append(float(np.mean((lvl[te] < thr).astype(int) == lab[te])))
        acc["trend"].append(float(np.mean((prev[te] > 0).astype(int) == lab[te])))
        maj = int(np.mean(lab[tr]) >= 0.5)
        acc["majority"].append(float(np.mean(lab[te] == maj)))
    out = {}
    for k, v in acc.items():
        a = np.asarray(v)
        h = sps.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        out[k] = (float(a.mean()), float(h))
    return out


def run(name, sources, signal_key):
    dev, nxt, prev, lab, blk, lvl, who = points(sources, signal_key)
    r, level, trend = stats(dev, nxt, prev, lab)
    ci = boot_ci(dev, nxt, prev, lab, blk)
    majority = float(max(np.mean(lab == 0), np.mean(lab == 1)))
    print(f"\n### {name} ({signal_key}) | n={len(lab)} | majority baseline {majority:.3f}")
    print(f"  lag-1 reversion (dev vs next change): {r:+.3f}  95% CI [{ci['r'][0]:+.3f}, {ci['r'][1]:+.3f}]")
    print(f"  level rule accuracy                 : {level:.3f}  95% CI [{ci['level'][0]:.3f}, {ci['level'][1]:.3f}]")
    print(f"  trend rule accuracy                 : {trend:.3f}  95% CI [{ci['trend'][0]:.3f}, {ci['trend'][1]:.3f}]")
    bl = blocked_rules(lvl, prev, lab, blk, who)
    print(f"  under the leakage-controlled folds, threshold from training blocks only:")
    for k in ("level", "trend", "majority"):
        m, h = bl[k]
        print(f"    {k:8s} {m:.3f} +/- {h:.3f}")
    per_source = {s: float(np.corrcoef(x[:-1] - x.mean(), np.diff(x))[0, 1])
                  for s, x in series(sources, signal_key).items() if len(x) > 3}
    print(f"  per source                          : "
          + ", ".join(f"{s} {v:+.3f}" for s, v in per_source.items()))
    return {"domain": name, "signal": signal_key, "n": int(len(lab)),
            "majority_baseline": round(majority, 4),
            "lag1_reversion": round(r, 4), "lag1_reversion_ci": [round(c, 4) for c in ci["r"]],
            "level_rule": round(level, 4), "level_rule_ci": [round(c, 4) for c in ci["level"]],
            "trend_rule": round(trend, 4), "trend_rule_ci": [round(c, 4) for c in ci["trend"]],
            "per_source_lag1": {s: round(v, 4) for s, v in per_source.items()},
            "blocked": {k: [round(v[0], 4), round(v[1], 4)] for k, v in bl.items()},
            "bootstrap": {"resamples": BOOT, "unit": "time block", "blocks": int(N_BLOCKS)}}


def main():
    res = [run("Dynamic graphs", FC.graph_sources(), "nodes"),
           run("Ecology (EcoMon)", FC.ecology_sources(), "richness")]
    json.dump(res, open(os.path.join(HERE, "results_mean_reversion.json"), "w"), indent=2)
    print("\nsaved results_mean_reversion.json")


if __name__ == "__main__":
    main()
