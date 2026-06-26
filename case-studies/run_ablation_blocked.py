"""Leakage-controlled ablation of the spatial-dispersion family + FDR selection.
Uses the same blocked CV as run_rigorous (StratifiedGroupKFold over time blocks for
graphs/ecology; StratifiedKFold for the independent memory episodes)."""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.feature_selection import SelectFdr, f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["common", "memory-sdn", "graphs-dynamic", "ecology-ecomon"]:
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import real_memory_demo as memory_demo, graphs_demo, ecology_demo
from pipeline import featurize_signals
from run_rigorous import make_cv, cv_acc

DISP = {
    "Memory (real)": {"spatial_entropy", "centroid"},
    "Dynamic graphs": {"density", "new_node_rate"},
    "Ecology (EcoMon)": {"turnover", "new_taxa_rate"},
}


def subset(s, keep=None, drop=None):
    if keep is not None:
        return {k: v for k, v in s.items() if k in keep}
    return {k: v for k, v in s.items() if k not in drop}


def run(name, sigs, y, blocks):
    y = np.asarray(y)
    cv, groups = make_cv(blocks)
    disp = DISP[name]
    Xfull = np.array([featurize_signals(s) for s in sigs])
    Xnod = np.array([featurize_signals(subset(s, drop=disp)) for s in sigs])
    Xdis = np.array([featurize_signals(subset(s, keep=disp)) for s in sigs])
    full, _ = cv_acc(Xfull, y, cv, groups)
    nod, _ = cv_acc(Xnod, y, cv, groups)
    dis, _ = cv_acc(Xdis, y, cv, groups)
    sel = SelectFdr(f_classif, alpha=0.05).fit(np.nan_to_num(Xfull), y)
    nsel = int(sel.get_support().sum())
    fdr, _ = cv_acc(sel.transform(np.nan_to_num(Xfull)), y, cv, groups)
    out = {"domain": name, "dispersion_family": sorted(disp),
           "full": round(full, 4), "minus_dispersion": round(nod, 4),
           "dispersion_gain": round(full - nod, 4), "dispersion_only": round(dis, 4),
           "fdr_kept": nsel, "fdr_total": int(Xfull.shape[1]), "fdr_acc": round(fdr, 4)}
    print(f"{name}: full={full:.3f} minus_disp={nod:.3f} gain={full-nod:+.3f} "
          f"disp_only={dis:.3f} FDR={nsel}/{Xfull.shape[1]}->{fdr:.3f}")
    return out


def main():
    res = []
    sm, ym = memory_demo.build_signals()
    res.append(run("Memory (real)", sm, ym, None))
    sg, yg, bg = graphs_demo.build_signals_blocked()
    res.append(run("Dynamic graphs", sg, yg, bg))
    samples = ecology_demo.load_samples()
    se, ye, be = ecology_demo.build_signals_blocked(samples)
    res.append(run("Ecology (EcoMon)", se, ye, be))
    json.dump(res, open(os.path.join(HERE, "results_ablation_blocked.json"), "w"), indent=2)
    print("saved results_ablation_blocked.json")


if __name__ == "__main__":
    main()
