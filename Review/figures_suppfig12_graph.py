"""Supplementary Figure 12: the index by age band, and the graph diagnostics.

Figure 6 shows the index against continuous age, which electrodes carry it,
and its replication per corpus. This supplement adds what does not fit there:

Panel order follows the order the Results cite them, so a reader never meets
a later letter first:

  A  the learned graph index by age band
  B  every graph measure, for the marginal and the conditional graph
  C  the per-edge picture behind the node-strength summary
  D  the raw-covariance control
  E  the spectral index by age band
  F  the two combined

USAGE: python figures_suppfig12_graph.py
"""
import argparse
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

from graph_analysis import to_corr, to_partial, metrics, CH, ANT, POS  # noqa: E402
from pasa_age_groups import perm_p, spearman, paper_bands, _letter  # noqa: E402
from pasa_index import _pearson, holm  # noqa: E402
from figures_pasa_logdiff_test import panel_d  # noqa: E402

# paper palette
NAVY, RED, PURPLE = "#1F4E79", "#C00000", "#7030a0"
TEAL, GREY = "#1B7F79", "#808080"
NPERM = 10000


def ep(r, n):
    t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-18))
    return 2 * stats.t.sf(abs(t), n - 2)


def strength(W):
    return np.abs(W - np.eye(14)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()

    c = np.load(os.path.join(RES, "pasa_cohorts_features.npz"), allow_pickle=True)
    ds, age_all = c["dataset"].astype(str), c["age"]
    m = ds == "ours"
    subs, age = c["subject"][m].astype(str), c["age"][m]

    Wm_all = np.stack([to_corr(expm(G)) for G in c["glog"]])
    S_all = np.stack([strength(w) for w in Wm_all])
    graph = S_all[m][:, ANT].mean(1) - S_all[m][:, POS].mean(1)

    p = np.load(os.path.join(RES, "pasa_power_table.npz"), allow_pickle=True)
    assert list(p["uniq"].astype(str)) == list(subs), "subject order differs"
    _, BS, _ = paper_bands()
    Pw = {k: p["P_" + k] for k in p["pkeys"]}
    bs = np.mean([Pw["abs_" + b] for b in BS if "abs_" + b in Pw], axis=0)
    fft = bs[:, ANT].mean(1) - bs[:, POS].mean(1)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    comb = (zs(graph) + zs(fft)) / 2.0

    fig, axs = plt.subplots(2, 3, figsize=(16.8, 10.4))
    axs = axs.ravel()
    rows = []

    # ---- A, E, F  three in-house indices, by age band ------------------
    for slot, (v, ttl, ylab, col, note, lett) in zip((0, 4, 5), (
            (graph, "A. Learned G$_{log}$ graph, anterior minus posterior "
                    "node strength", "ANT - POS node strength", NAVY,
             "model arm; all 81 subjects", "A"),
            (fft, "E. FFT brain-score bands (beta + gamma)",
             "ANT - POS  (log$_{10}$ power)", RED,
             "model-free; all 81 subjects", "E"),
            (comb, "F. Combined index, z(graph) + z(FFT)",
             "combined z score", PURPLE,
             "equal weights, fixed a priori", "F"))):
        pp = perm_p(v, age, np.random.default_rng(11), NPERM)
        gs, r, rs = panel_d(axs[slot], v, age, ttl, ylab, col, note,
                            letter=lett, pperm=pp)
        rows.append(dict(panel=lett, index=ttl[3:].replace("$_{log}$", "_log"),
                         n=len(age), r=round(r, 4), rho=round(rs, 4),
                         p_perm=round(pp, 4),
                         **{g: round(mu, 4) for g, mu in
                            zip(gs.group, gs["mean"])}))

    # ---- C  per-edge correlation with age ------------------------------
    ax = axs[2]
    off = ~np.eye(14, dtype=bool)
    R = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if i != j:
                R[i, j] = _pearson(Wm_all[:, i, j], age_all)
    v = np.nanmax(np.abs(R[off]))
    im = ax.imshow(np.where(off, R, np.nan), cmap="RdBu_r", vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_xticks(range(14)); ax.set_yticks(range(14))
    ax.set_xticklabels(CH, rotation=90, fontsize=6.5)
    ax.set_yticklabels(CH, fontsize=6.5)
    nsig = int((np.abs(R[np.triu_indices(14, 1)]) >
                1.96 / np.sqrt(len(age_all) - 3)).sum())
    ax.set_title(f"C. Per-edge correlation with age (n = {len(age_all)})\n"
                 f"{nsig} of 91 edges exceed the uncorrected threshold\n"
                 f"(4.5 expected by chance)", fontsize=9.5, loc="left")
    _letter(ax, "C")

    # ---- B  every measure, marginal vs conditional (seperate columns) ----
    ax = axs[1]
    Wc = np.stack([to_partial(expm(G)) for G in c["glog"]])
    Fm = pd.DataFrame([metrics(w) for w in Wm_all])
    Fc = pd.DataFrame([metrics(w) for w in Wc])
    METS = ["strength_ANT_minus_POS", "clustering", "strength_POS",
            "strength_ANT", "efficiency", "path_length", "modularity",
            "strength"]
    SHORT = {"strength_ANT_minus_POS": "ANT - POS strength",
             "strength_ANT": "anterior strength",
             "strength_POS": "posterior strength",
             "strength": "total strength", "path_length": "path length"}
    rm = np.array([_pearson(Fm[k].to_numpy(), age_all) for k in METS])
    rc = np.array([_pearson(Fc[k].to_numpy(), age_all) for k in METS])
    hm = holm(np.array([ep(x, len(age_all)) for x in rm]))
    hc = holm(np.array([ep(x, len(age_all)) for x in rc]))
    y = np.arange(len(METS))
    ax.barh(y + 0.19, rm, height=.36, color=NAVY, edgecolor="k", lw=.4,
            label="marginal (correlation)")
    ax.barh(y - 0.19, rc, height=.36, color=GREY, edgecolor="k", lw=.4,
            label="conditional (partial correlation)")
    for i in range(len(METS)):
        for rv, hv, yo, cl in ((rm[i], hm[i], 0.19, NAVY),
                               (rc[i], hc[i], -0.19, "#555555")):
            if hv < 0.05:
                ax.text(rv + (0.012 if rv > 0 else -0.012), y[i] + yo, "*",
                        va="center", ha="left" if rv > 0 else "right",
                        fontsize=11, color=cl)
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(k, k) for k in METS],
                                         fontsize=8)
    ax.axvline(0, color="k", lw=1); ax.invert_yaxis()
    ax.set_xlabel("r with age", fontsize=8.5)
    ax.set_title("B. Every graph measure, both notions of connection\n"
                 "* = Holm p < 0.05 within that graph; the conditional\n"
                 "graph carries almost nothing", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    _letter(ax, "B")

    # ---- D  raw-covariance control --------------------------------------
    ax = axs[3]
    Sr = np.stack([strength(to_corr(expm(G))) for G in c["logcov"]])
    ir = Sr[:, ANT].mean(1) - Sr[:, POS].mean(1)
    ax.scatter(age_all, ir, s=10, c=GREY, alpha=.5, lw=0)
    b1, b0 = np.polyfit(age_all, ir, 1)
    xs = np.array([age_all.min(), age_all.max()])
    ax.plot(xs, b1 * xs + b0, color="k", lw=2, ls="--")
    rr = _pearson(ir, age_all)
    ax.set_title(f"D. Control: same index, RAW covariance graph\n"
                 f"n = {len(age_all)}, r = {rr:+.3f}, "
                 f"p = {ep(rr, len(ir)):.3f};\n"
                 f"the shift is contributed by the model",
                 fontsize=9.5, loc="left")
    ax.set_xlabel("chronological age (years)", fontsize=8.5)
    ax.set_ylabel("ANT - POS node strength", fontsize=8.5)
    _letter(ax, "D")

    for ax in axs:
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"SuppFig12.{ext}"), dpi=165,
                    bbox_inches="tight")
    print("wrote Figures/SuppFig12.pdf (+ .png)")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "Table_suppfig12_panels.csv"), index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nedges above the uncorrected threshold: {nsig} of 91")
    print(f"raw-covariance control: r = {rr:+.3f}, p = {ep(rr, len(ir)):.4f}")


if __name__ == "__main__":
    main()
