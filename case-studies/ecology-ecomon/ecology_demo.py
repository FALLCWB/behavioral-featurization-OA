"""Ecology case study for Behavioral Featurization (EcoMon plankton, 1977-2015).

Each plankton tow is a community sample; the set of taxa present (count > 0) is
volatile and unbounded, changing from one survey to the next, and is not known a
priori (the both-at-once condition). Successive samples within a region form an
evolving community. The same shared recipe derives behavioral signals from that
evolution (richness, abundance, diversity, taxon turnover, novelty) and classifies
which region a window of consecutive samples came from.

Identical pipeline to the memory and dynamic-graph case studies
(case-studies/common/pipeline.py): reusing the same code is the generality argument.

Data: BCO-DMO dataset 3327, plankton10m2_v3_1_sort.csv (long format:
one row per sample x taxon, with an abundance 'count').
"""
from __future__ import annotations
import os
import sys
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pipeline import featurize_signals, run_classification

CSV = os.path.join(os.path.dirname(__file__), "data", "ecomon.csv")
WIN = 20            # consecutive samples per window
TARGET_PER_REGION = 300


def region_of(lat):
    if lat < 39.5:
        return "south"
    if lat < 42.0:
        return "mid"
    return "north"


def load_samples():
    """Aggregate the long-format CSV into per-sample community summaries."""
    taxa_id = {}
    samples = {}   # key -> dict
    with open(CSV, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col = {name: i for i, name in enumerate(header)}
        ci_date, ci_cr, ci_st = col["date"], col["cruiseid"], col["station"]
        ci_lat, ci_taxon, ci_count = col["lat"], col["taxon"], col["count"]
        for row in r:
            cnt = row[ci_count]
            if cnt == "nd":
                continue
            try:
                c = float(cnt)
            except ValueError:
                continue
            if c <= 0:
                continue
            key = (row[ci_date], row[ci_cr], row[ci_st])
            s = samples.get(key)
            if s is None:
                try:
                    lat = float(row[ci_lat])
                except ValueError:
                    continue
                s = {"date": row[ci_date], "lat": lat, "taxa": set(), "N": 0.0, "S": 0.0}
                samples[key] = s
            tx = row[ci_taxon]
            tid = taxa_id.get(tx)
            if tid is None:
                tid = len(taxa_id)
                taxa_id[tx] = tid
            s["taxa"].add(tid)
            s["N"] += c
            s["S"] += c * math.log(c)
    return list(samples.values())


def window_signals(win):
    """Behavioral signals over a window of consecutive community samples."""
    sig = {k: [] for k in ["richness", "log_abund", "diversity", "turnover", "new_taxa_rate"]}
    seen = set()
    prev = None
    for s in win:
        taxa = s["taxa"]
        N = s["N"]
        rich = len(taxa)
        # Shannon diversity: H = ln(N) - (1/N) * sum n_i ln n_i
        H = (math.log(N) - s["S"] / N) if N > 0 else 0.0
        turnover = len(taxa ^ prev) if prev is not None else 0
        new_rate = len(taxa - seen) / max(1, rich)
        seen |= taxa
        prev = taxa
        sig["richness"].append(rich)
        sig["log_abund"].append(math.log1p(N))
        sig["diversity"].append(H)
        sig["turnover"].append(turnover)
        sig["new_taxa_rate"].append(new_rate)
    return sig


def build_signals_blocked(samples, n_blocks=5):
    """Leakage-free version: NON-overlapping windows per region, with a time-block id
    so cross-validation can hold out whole contiguous time blocks of each region."""
    by_region = {"south": [], "mid": [], "north": []}
    for s in samples:
        by_region[region_of(s["lat"])].append(s)
    sigs, y, block = [], [], []
    for region, lst in by_region.items():
        lst.sort(key=lambda s: s["date"])
        nwin = len(lst) // WIN
        for i in range(nwin):
            sigs.append(window_signals(lst[i * WIN:(i + 1) * WIN]))
            y.append(region)
            block.append(min(n_blocks - 1, i * n_blocks // nwin))
    return sigs, np.array(y), np.array(block)


def build_signals(samples):
    """Raw per-window behavioral signals (for the contrast / ablation harness)."""
    by_region = {"south": [], "mid": [], "north": []}
    for s in samples:
        by_region[region_of(s["lat"])].append(s)
    sigs, y = [], []
    for region, lst in by_region.items():
        lst.sort(key=lambda s: s["date"])
        n = len(lst)
        step = max(1, (n - WIN) // TARGET_PER_REGION)
        i = 0
        while i + WIN <= n:
            sigs.append(window_signals(lst[i:i + WIN]))
            y.append(region)
            i += step
    return sigs, np.array(y)


def build_dataset(samples):
    by_region = {"south": [], "mid": [], "north": []}
    for s in samples:
        by_region[region_of(s["lat"])].append(s)
    X, y, reps = [], [], {}
    counts = {}
    for region, lst in by_region.items():
        lst.sort(key=lambda s: s["date"])
        n = len(lst)
        step = max(1, (n - WIN) // TARGET_PER_REGION)
        c = 0
        i = 0
        first = True
        while i + WIN <= n:
            sig = window_signals(lst[i:i + WIN])
            X.append(featurize_signals(sig))
            y.append(region)
            if first:
                reps[region] = sig
                first = False
            i += step
            c += 1
        counts[region] = c
    return np.array(X), np.array(y), reps, counts


def make_figure(reps, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"south": "tab:orange", "mid": "tab:blue", "north": "tab:green"}
    for region, sig in reps.items():
        axes[0].plot(sig["richness"], label=region, color=colors[region], lw=1.8)
        axes[1].plot(sig["turnover"], label=region, color=colors[region], lw=1.8)
    axes[0].set_xlabel("sample step within window")
    axes[0].set_ylabel("taxon richness")
    axes[0].set_title("Evolving community: taxon richness over time, by region")
    axes[0].legend()
    axes[1].set_xlabel("sample step within window")
    axes[1].set_ylabel("taxon turnover")
    axes[1].set_title("Community turnover over time, by region")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    here = os.path.dirname(__file__)
    figdir = os.path.join(here, "..", "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    print("loading samples (streaming the long-format CSV)...")
    samples = load_samples()
    print(f"unique community samples: {len(samples)}")
    X, y, reps, counts = build_dataset(samples)
    res = run_classification(X, y)
    print("=== Behavioral Featurization - ecology (EcoMon plankton communities) ===")
    print(f"windows per region: {counts}")
    print(f"total windows: {len(y)} | classes: {res['labels']} | features per object: {X.shape[1]}")
    print(f"CV accuracy: {res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")
    print(f"confusion (rows=true {res['labels']}, cols=pred): {res['confusion']}")
    print(res["report"])
    figpath = os.path.join(figdir, "F4b_ecology_demo.png")
    make_figure(reps, figpath)
    print("figure saved:", os.path.relpath(figpath, here))
    with open(os.path.join(here, "results_ecology.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)


if __name__ == "__main__":
    main()
