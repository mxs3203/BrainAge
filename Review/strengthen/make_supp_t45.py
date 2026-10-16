"""Rebuild Supplementary Tables 4 and 5 on what the paper now reports.

Table 4 was on the exp(d) ratio scale and listed the raw log-covariance arm,
which Figure 6 no longer shows. It is rebuilt on the log-difference scale
with the model-graph arm in place of the log-covariance arm.

Table 5 carried two diagonal columns that are cited nowhere; they are
replaced by the graph node strength, which the Discussion does cite.

USAGE: python make_supp_t45.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REV)
RES = os.path.join(REV, "results")

from pasa_age_groups import IX, CH, perm_p, spearman, paper_bands  # noqa: E402
from pasa_index import _pearson, holm  # noqa: E402
from graph_analysis import to_corr  # noqa: E402

ANT, POS = IX["ANT"], IX["POS"]
NPERM = 10000


def main():
    c = np.load(os.path.join(RES, "pasa_cohorts_features.npz"), allow_pickle=True)
    m = c["dataset"].astype(str) == "ours"
    subs, age = c["subject"][m].astype(str), c["age"][m]
    S = np.stack([np.abs(to_corr(expm(G)) - np.eye(14)).sum(1)
                  for G in c["glog"][m]])
    graph = S[:, ANT].mean(1) - S[:, POS].mean(1)

    p = np.load(os.path.join(RES, "pasa_power_table.npz"), allow_pickle=True)
    assert list(p["uniq"].astype(str)) == list(subs)
    _, BS, _ = paper_bands()
    Pw = {k: p["P_" + k] for k in p["pkeys"]}
    bs = np.mean([Pw["abs_" + b] for b in BS if "abs_" + b in Pw], axis=0)

    def d(P, ia=ANT):
        return P[:, ia].mean(1) - P[:, POS].mean(1)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    fft = d(bs)
    comb = (zs(graph) + zs(fft)) / 2.0

    rows, band_p = [], []
    def add(arm, name, v):
        pp = perm_p(v, age, np.random.default_rng(11), NPERM)
        rows.append(dict(arm=arm, index=name, n=len(age),
                         r=round(_pearson(v, age), 4),
                         spearman=round(spearman(v, age), 4),
                         p_perm=round(pp, 4)))
        return pp

    add("model", "graph ANT - POS node strength", graph)
    add("spectral", "FFT brain-score bands (beta + gamma)", fft)
    add("combined", "z(graph) + z(FFT), equal weights", comb)
    for b in ("delta", "theta", "alpha", "beta", "gamma"):
        band_p.append(add("spectral", f"FFT {b} ANT - POS", d(Pw["abs_" + b])))
    SUB = {"prefrontal AF3/AF4 over posterior": ["AF3", "AF4"],
           "lateral frontal F7 F3 F4 F8 over posterior": ["F7", "F3", "F4", "F8"],
           "frontocentral FC5/FC6 over posterior": ["FC5", "FC6"]}
    for nm, chs in SUB.items():
        add("spectral", nm, d(bs, np.array([CH.index(x) for x in chs])))

    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(RES, "SuppT4_pasa_indices.csv"), index=False)
    hb = holm(np.array(band_p))
    print("Supplementary Table 4")
    print(t4.to_string(index=False))
    print(f"\n  beta Holm across the five bands = {hb[3]:.4f}")

    # ---- Table 5: drop the two diagonals nothing cites -----------------
    t5 = pd.read_csv(os.path.join(RES, "Table_lobe_per_channel.csv"))
    t5 = t5.drop(columns=[x for x in ("r_glog_diag", "r_logcov_diag")
                          if x in t5.columns])
    # deliberately no node-strength column: the per-electrode graph numbers
    # are n=799 and the band columns are n=81, side by side in one table that
    # just invites a comparsion that isn't there. They're in SuppFig12C.
    cols = ["channel", "region"] + \
           [x for x in t5.columns if x.startswith("r_log_")]
    t5[cols].to_csv(os.path.join(RES, "SuppT5_per_channel.csv"), index=False)
    print("\nSupplementary Table 5")
    print(t5[cols].to_string(index=False))


if __name__ == "__main__":
    main()
