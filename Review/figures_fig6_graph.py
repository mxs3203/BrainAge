"""Figure 6. Anterior shift in the model's learned graph.

Strength comes off the LEARNED G_log, not the raw signal:

    G_log -> C = expm(G_log) -> R_ij = C_ij / sqrt(C_ii C_jj)
    strength_i = sum_{j != i} |R_ij|
    index      = mean(strength, 8 ant) - mean(strength, 4 pos)

The correlation-normalisation divides out per-channel power, which is why this
picks up the shift and the G_log diagonal doesn't.

USAGE: python figures_fig6_graph.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "Figures")

from graph_analysis import to_corr, metrics, CH, ANT, POS  # noqa: E402
from pasa_age_groups import perm_p, spearman, _letter  # noqa: E402
from pasa_index import _pearson, holm  # noqa: E402

NPERM = 100000
# paper palette. corpus colours match Figure 7, ant/pos stay red/navy like
# everywhere else in teh paper
NAVY, RED, PURPLE = "#1F4E79", "#C00000", "#6A3D9A"
TEAL, GREY = "#1B7F79", "#808080"
COL = {"ours": PURPLE, "ds003775": NAVY, "ds005385": TEAL}
NAME = {"ours": "in-house", "ds003775": "ds003775", "ds005385": "ds005385"}
REGION = {"anterior": RED, "posterior": NAVY, "temporal": GREY}


def exact_p(r, n):
    t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-18))
    return 2 * stats.t.sf(abs(t), n - 2)


def main():
    c = np.load(os.path.join(RES, "pasa_cohorts_features.npz"), allow_pickle=True)
    ds, age = c["dataset"].astype(str), c["age"]
    W = np.stack([to_corr(expm(G)) for G in c["glog"]])
    v = pd.DataFrame([metrics(w) for w in W])["strength_ANT_minus_POS"].to_numpy()
    S = np.stack([np.abs(w - np.eye(14)).sum(1) for w in W])   # node strength

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))

    # ---- A  index vs age, all three corpora -----------------------------
    ax = axes[0]
    for d in ("ds005385", "ds003775", "ours"):
        m = ds == d
        ax.scatter(age[m], v[m], s=14, c=COL[d], alpha=.45, lw=0,
                   label=f"{NAME[d]} (n={int(m.sum())})  "
                         f"r = {_pearson(v[m], age[m]):+.3f}, "
                         f"p = {exact_p(_pearson(v[m], age[m]), int(m.sum())):.3g}")
    b, a0 = np.polyfit(age, v, 1)
    xs = np.array([age.min(), age.max()])
    ax.plot(xs, b * xs + a0, color="k", lw=2.4, zorder=5)
    r = _pearson(v, age)
    pp = perm_p(v, age, np.random.default_rng(3), NPERM)
    ax.set_title("A. Anterior minus posterior node strength, learned "
                 "G$_{log}$ graph\n"
                 f"all three corpora: n = {len(age)}, r = {r:+.3f}, "
                 f"$\\rho$ = {spearman(v, age):+.3f}, "
                 f"p = {exact_p(r, len(age)):.1e}", fontsize=10, loc="left")
    ax.set_xlabel("chronological age (years)", fontsize=9.5)
    ax.set_ylabel("ANT - POS node strength", fontsize=9.5)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8.5)
    _letter(ax, "A")
    print(f"A  all799 r={r:+.3f} perm p={pp:.6f} ({NPERM} permutations)")
    # ---- C  which electrodes carry it ----------------------------------
    ax = axes[2]
    rn = np.array([_pearson(S[:, i], age) for i in range(14)])
    order = np.argsort(rn)
    aset, pset = set(ANT.tolist()), set(POS.tolist())
    cols = [REGION["anterior"] if i in aset else
            REGION["posterior"] if i in pset else REGION["temporal"]
            for i in order]
    hb = holm(np.array([exact_p(x, len(age)) for x in rn]))
    ax.barh(range(14), rn[order], color=cols, edgecolor="k", lw=.4)
    ax.set_yticks(range(14))
    ax.set_yticklabels([CH[i] for i in order], fontsize=8.5)
    ax.axvline(0, color="k", lw=1)
    crit = 1.96 / np.sqrt(len(age) - 3)
    for s_ in (-1, 1):
        ax.axvline(s_ * crit, color="#555555", ls=":", lw=1.1,
                   label="p = 0.05 (uncorrected)" if s_ > 0 else None)
    ax.set_xlabel("r of that electrode's node strength with age", fontsize=9.5)
    ax.set_title("C. Per-electrode node strength against age (n = 799)\n"
                 f"red = anterior, navy = posterior, grey = temporal; "
                 f"all {int((hb < 0.05).sum())}/14 survive Holm",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8.5)
    _letter(ax, "C")

    # ---- B  same index, per corpus ---------------------------------------
    ax = axes[1]
    tab = pd.read_csv(os.path.join(RES, "Table_graph_bydataset.csv"))
    sel = tab[(tab.source == "model G_log") & (tab.graph == "marginal") &
              (tab.metric == "strength_ANT_minus_POS")]
    ROWS = [("all three corpora", "all 799", "#2B2B2B"),
            ("dataset-centred", "centred 799", "#6E6E6E"),
            ("ds005385", "ds005385", COL["ds005385"]),
            ("ds003775", "ds003775", COL["ds003775"]),
            ("in-house", "in-house", COL["ours"])]
    praw = np.array([exact_p(float(sel[sel.cohort == k].r.iloc[0]),
                             int(sel[sel.cohort == k].n.iloc[0]))
                     for _, k, _ in ROWS])
    pholm = holm(praw)
    ypos, ylab = [], []
    for k, (lab, key, col) in enumerate(ROWS):
        row = sel[sel.cohort == key]
        if not len(row):
            continue
        rr, nn = float(row.r.iloc[0]), int(row.n.iloc[0])
        pe = pholm[k]
        se = 1 / np.sqrt(max(nn - 3, 1))
        zf = np.arctanh(np.clip(rr, -.999, .999))
        lo, hi = np.tanh(zf - 1.96 * se), np.tanh(zf + 1.96 * se)
        ax.plot([lo, hi], [-k, -k], color=col, lw=2.0, alpha=.9)
        ax.plot(rr, -k, "o", ms=9, color=col,
                mfc=col if pe < 0.05 else "white", mew=1.8)
        ax.text(hi + 0.02, -k,
                f"r = {rr:+.3f},  p = {praw[k]:.3g}   (Holm {pe:.3g})",
                va="center", fontsize=7.8, color=col)
        ypos.append(-k); ylab.append(f"{lab}  (n = {nn})")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=9)
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel("r with age (95% CI); filled = Holm p < 0.05", fontsize=9.5)
    ax.set_title("B. The same index in each corpus separately\n"
                 "all five survive Holm correction across the five tests",
                 fontsize=10, loc="left")
    ax.tick_params(labelsize=8.5)
    _letter(ax, "B")

    for a_ in axes:
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig6.{ext}"), dpi=165,
                    bbox_inches="tight")
    print("wrote Figures/Fig6.pdf (+ .png)")

    out = pd.DataFrame([dict(electrode=CH[i], region=("anterior" if i in aset
                        else "posterior" if i in pset else "temporal"),
                        r=round(rn[i], 4), p=float(f"{exact_p(rn[i], len(age)):.3g}"))
                        for i in range(14)]).sort_values("r")
    out.to_csv(os.path.join(RES, "Table_graph_node_strength.csv"), index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
