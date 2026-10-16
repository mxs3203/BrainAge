"""Figure 6 and Supplementary Figure 12 on the log-difference scale.

The published versions plot exp(d), where

    d = mean over anterior channels (v_c) - mean over posterior channels (v_c)

and v_c is already a log-scale quantity: log10 band power for the FFT arm, and
the diagonal of the matrix logarithm for the two covariance arms. Exponentiating
reintroduces the skew the logarithm removed, and for the FFT arm it does not
even produce the ratio, because the input is log10 while exp() undoes a natural
log: exp(d) is (ANT/POS)^0.434.

This renders both figures on d itself and labels each axis for what it holds.
Outputs carry a _test suffix and do not overwrite the manuscript figures.

USAGE: python figures_pasa_logdiff_test.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "Figures")

from pasa_age_groups import (  # noqa: E402
    IX, group_stats, _letter, perm_p, spearman, paper_bands, NAVY, RED, GREY)
from pasa_index import _pearson  # noqa: E402

ANT, POS = IX["ANT"], IX["POS"]
EDGES = [20, 30, 40, 50, 63]
LABELS = ["20-29", "30-39", "40-49", "50-62"]
PURPLE = "#7030a0"


AGG = {"mean": np.mean, "median": np.median}


def diff(V, agg="mean"):
    """d: anterior minus posterior, aggregated across the channels of each
    group. Both inputs are already on a log scale, so this is the log of a
    ratio of geometric means (agg="mean") or of medians (agg="median").

    The median is the robust version: with 8 anterior and 4 posterior
    electrodes it stops one bad channel from carrying the group, at the cost
    of discarding most of the posterior group (the median of 4 is the mean of
    the middle 2).
    """
    f = AGG[agg]
    return np.asarray([f(v[ANT]) - f(v[POS]) for v in V], float)


def group_summary(v, age, stat):
    """Per age bin: mean +/- SEM, or median +/- bootstrap SE of the median."""
    gs, gi = group_stats(v, age, EDGES, LABELS)
    if stat == "mean":
        return gs, gi
    rng = np.random.default_rng(4)
    for g in range(len(LABELS)):
        m = gi == g
        n = int(m.sum())
        if n == 0:
            continue
        x = v[m]
        boot = np.median(rng.choice(x, size=(2000, n), replace=True), axis=1)
        gs.loc[g, "mean"] = float(np.median(x))
        gs.loc[g, "sem"] = float(boot.std(ddof=1)) if n > 1 else np.nan
    return gs, gi


def panel_d(ax, v, age, title, ylab, colour, note="", letter=None,
            pperm=None, ref=0.0, stat="mean"):
    """Age-group panel on a log-difference axis, referenced to d = 0."""
    gs, gi = group_summary(v, age, stat)
    rng = np.random.default_rng(7)
    x = gi + rng.uniform(-0.13, 0.13, len(gi))
    ax.scatter(x, v, s=13, c=colour, alpha=0.35, linewidths=0, zorder=2)
    ok = gs.n > 0
    ax.errorbar(np.flatnonzero(ok), gs.loc[ok, "mean"], yerr=gs.loc[ok, "sem"],
                fmt="o-", color=colour, lw=2.0, ms=7, capsize=4, zorder=4,
                mfc="white", mew=2.0)
    ax.axhline(ref, color=GREY, lw=0.8, ls=":", zorder=1)
    r, rs = _pearson(v, age), spearman(v, age)
    ax.set_title(title, fontsize=9.5, loc="left")
    if letter:
        _letter(ax, letter)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([f"{l}\nn={int(n)}" for l, n in zip(LABELS, gs.n)],
                       fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    lab = f"continuous age: r={r:+.3f}  rho={rs:+.3f}"
    if pperm is not None:
        lab += f"  p={pperm:.3f}" + ("*" if pperm < 0.05 else " (n.s.)")
    ax.text(0.015, 0.025, lab + (f"\n{note}" if note else ""),
            transform=ax.transAxes, fontsize=7.2, va="bottom", color="#333333")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return gs, r, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--stat", default="mean", choices=["mean", "median"],
                    help="how channels are aggregated within anterior and "
                         "posterior, and how each age bin is summarised")
    a = ap.parse_args()
    SUF = "_test" if a.stat == "mean" else "_test_median"
    print(f"aggregating channels and age bins by {a.stat}")

    _, BS_BANDS, _ = paper_bands()
    z = np.load(os.path.join(RES, "pasa_power_table.npz"), allow_pickle=True)
    age = z["age"]
    Pw = {k: z["P_" + k] for k in z["pkeys"]}
    d_cov = diff([np.diag(G) for G in z["logcov"]], a.stat)
    bs = np.mean([Pw["abs_" + b] for b in BS_BANDS if "abs_" + b in Pw], axis=0)
    d_bs = diff(bs, a.stat)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    d_comb = (zs(d_cov) + zs(d_bs)) / 2.0

    rows = []

    # ===================== FIGURE 6 (test) ===========================
    fig = plt.figure(figsize=(16.0, 5.1))
    gsp = fig.add_gridspec(1, 3, wspace=0.27)
    for k, (v, ttl, ylab, col, note, lett) in enumerate((
            (d_cov, "A. Raw log-covariance, anterior minus posterior",
             "ANT - POS  (log-Euclidean)", RED,
             "model-free; all 81 subjects", "A"),
            (d_bs, "B. FFT brain-score bands (beta + gamma)",
             "ANT - POS  (log$_{10}$ power)", RED,
             "model-free; all 81 subjects", "B"),
            (d_comb, "C. Combined index, z(log-cov) + z(FFT)",
             "combined z score", PURPLE,
             "equal weights, fixed a priori", "C"))):
        ax = fig.add_subplot(gsp[0, k])
        pp = perm_p(v, age, np.random.default_rng(11), a.nperm)
        gs, r, rs = panel_d(ax, v, age, ttl, ylab, col, note,
                            letter=lett, pperm=pp, stat=a.stat)
        rows.append(dict(figure="Fig6" + SUF, panel=lett, index=ttl[3:],
                         r=round(r, 4), rho=round(rs, 4), p_perm=round(pp, 4),
                         **{g: round(m, 4) for g, m in
                            zip(gs.group, gs["mean"])}))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig6{SUF}.{ext}"), dpi=165,
                    bbox_inches="tight")
    print(f"wrote Figures/Fig6{SUF}.pdf (+ .png)")

    # ===================== SUPPLEMENT 12 (test) ======================
    figS, axs = plt.subplots(1, 3, figsize=(15.0, 5.2))

    axf = axs[0]
    cmap = plt.get_cmap("viridis")
    keys = [b for b in ("delta", "theta", "alpha", "beta", "gamma")
            if "abs_" + b in Pw]
    for j, b in enumerate(keys):
        v = diff(Pw["abs_" + b], a.stat)
        gsx, _ = group_summary(v, age, a.stat)
        ok = gsx.n > 0
        r = _pearson(v, age)
        axf.errorbar(np.flatnonzero(ok), gsx.loc[ok, "mean"],
                     yerr=gsx.loc[ok, "sem"], fmt="o-", lw=1.6, ms=5,
                     capsize=3, color=cmap(j / max(1, len(keys) - 1)),
                     label=f"{b} r={r:+.3f}")
        rows.append(dict(figure="SuppFig12" + SUF, panel="A",
                         index=f"FFT {b} ANT-POS", r=round(r, 4),
                         rho=round(spearman(v, age), 4),
                         p_perm=round(perm_p(v, age, np.random.default_rng(11),
                                             a.nperm), 4),
                         **{g: round(m, 4) for g, m in
                            zip(gsx.group, gsx["mean"])}))
    axf.axhline(0.0, color=GREY, lw=0.8, ls=":")
    axf.set_title("A. FFT anterior minus posterior, per band",
                  fontsize=9.5, loc="left")
    _letter(axf, "A")
    axf.set_xticks(range(len(LABELS)))
    axf.set_xticklabels(LABELS, fontsize=7.5)
    axf.set_ylabel("ANT - POS  (log$_{10}$ power)", fontsize=8.5)
    axf.legend(fontsize=6.8, frameon=False, ncol=2)
    axf.tick_params(labelsize=7.5)

    axg = axs[1]
    axg.scatter(age, d_comb, s=26, c=RED, alpha=0.6, linewidths=0)
    b1, b0 = np.polyfit(age, d_comb, 1)
    xs = np.linspace(age.min(), age.max(), 50)
    axg.plot(xs, b1 * xs + b0, color=NAVY, lw=2)
    axg.set_title(f"B. Combined index vs continuous age  "
                  f"(r={_pearson(d_comb, age):+.3f})", fontsize=9.5, loc="left")
    _letter(axg, "B")
    axg.set_xlabel("age (years)", fontsize=8.5)
    axg.set_ylabel("combined z score", fontsize=8.5)
    axg.tick_params(labelsize=7.5)

    axh = axs[2]
    axh.scatter(zs(d_cov), zs(d_bs), s=26, c=PURPLE, alpha=0.6, linewidths=0)
    axh.set_title(f"C. Agreement between the two model-free arms  "
                  f"r={_pearson(d_cov, d_bs):+.3f}", fontsize=9.5, loc="left")
    _letter(axh, "C")
    axh.set_xlabel("z   log-covariance ANT - POS", fontsize=8.5)
    axh.set_ylabel("z   FFT brain-score ANT - POS", fontsize=8.5)
    axh.tick_params(labelsize=7.5)

    for ax in axs:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    figS.suptitle(
        "Supplement: composition and robustness of the frontal/posterior age "
        "index\nin-house EMOTIV cohort (n=81), model-free throughout; "
        f"log-difference scale, channels and bins aggregated by {a.stat}",
        fontsize=10.5, y=1.045)
    figS.tight_layout()
    for ext in ("pdf", "png"):
        figS.savefig(os.path.join(OUT, f"SuppFig12{SUF}.{ext}"), dpi=165,
                     bbox_inches="tight")
    print(f"wrote Figures/SuppFig12{SUF}.pdf (+ .png)")

    out = pd.DataFrame(rows)
    fn = os.path.join(RES, f"Table_pasa_logdiff{SUF}.csv")
    out.to_csv(fn, index=False)
    print("\n" + out.to_string(index=False))
    print("\nwrote", fn)


if __name__ == "__main__":
    main()
