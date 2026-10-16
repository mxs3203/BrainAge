"""Figures 6 and 7 and Supplementary Figure 14, regrouped.

Fig6      = the three model-free PASA arms.
Fig7      = the four predicted-vs-true panels.
SuppFig14 = the two G_log age panels + the per-tertile matrices and r histogram.
Nothing is recomputed from the model; everything comes from the saved tables.
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
    CH, IX, log_ratio, panel, _letter, perm_p, paper_bands, NAVY, RED)
from pasa_index import _pearson  # noqa: E402
import figures_pred_vs_true_4panel as P4  # noqa: E402

ANT, POS = IX["ANT"], IX["POS"]
EDGES = [20, 30, 40, 50, 63]
LABELS = ["20-29", "30-39", "40-49", "50-62"]


def load_glog(config):
    M = np.load(os.path.join(RES, f"glog_{config}_locked.npy"))
    age = np.load(os.path.join(RES, f"glog_{config}_locked_ages.npy"))
    q = np.quantile(age, [1 / 3, 2 / 3])
    grp = np.digitize(age, q)
    gm = [M[grp == k].mean(0) for k in range(3)]
    names = [f"young (<{q[0]:.0f}, n={int((grp==0).sum())})",
             f"mid ({q[0]:.0f}-{q[1]:.0f}, n={int((grp==1).sum())})",
             f"old (>{q[1]:.0f}, n={int((grp==2).sum())})"]
    R = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            v = M[:, i, j]
            R[i, j] = _pearson(v, age) if v.std() > 1e-12 else 0.0
    return M, age, gm, names, R


def load_arms():
    """The two model-free indices and their equal-weight combination."""
    _, BS_BANDS, _ = paper_bands()
    z = np.load(os.path.join(RES, "pasa_power_table.npz"), allow_pickle=True)
    age = z["age"]
    Lm = z["logcov"]
    Pw = {k: z["P_" + k] for k in z["pkeys"]}
    R_cov, d_cov = log_ratio([np.diag(G) for G in Lm], ANT, POS)
    bs = np.mean([Pw["abs_" + b] for b in BS_BANDS if "abs_" + b in Pw], axis=0)
    R_bs, d_bs = log_ratio(bs, ANT, POS)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    return age, R_cov, R_bs, (zs(d_cov) + zs(d_bs)) / 2.0


def mat(ax, A, title, letter, fig, vmin, vmax, cmap="RdBu_r", fs=9.5):
    off = ~np.eye(14, dtype=bool)
    im = ax.imshow(np.where(off, A, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fs, loc="left")
    fig.colorbar(im, ax=ax, fraction=.046)
    for s_ in (ax.set_xticks, ax.set_yticks):
        s_(range(14))
    ax.set_xticklabels(CH, rotation=90, fontsize=6)
    ax.set_yticklabels(CH, fontsize=6)
    _letter(ax, letter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--pred", default="age_phase4_t25_oldmodel_newdata")
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()

    M, age_m, gm, names, R = load_glog(a.config)
    age_r, R_cov, R_bs, R_comb = load_arms()
    df = pd.read_csv(os.path.join(RES, f"Table_pred_joint_{a.pred}.csv"))
    print(f"{len(age_m)} locked subjects, {len(age_r)} model-free subjects, "
          f"{len(df)} prediction rows")

    # ===================== FIGURE 6: the model-free arms =============
    fig = plt.figure(figsize=(16.0, 5.1))
    gs = fig.add_gridspec(1, 3, wspace=0.27)
    for k, (v, ttl, ylab, col, note, lett) in enumerate((
            (R_cov, "A. Raw log-covariance, anterior/posterior",
             "ANT / POS  (ratio)", RED, "model-free; all 81 subjects", "A"),
            (R_bs, "B. FFT brain-score bands (beta + gamma)",
             "ANT / POS power ratio", RED, "model-free; all 81 subjects", "B"),
            (R_comb, "C. Combined index, z(log-cov) + z(FFT)",
             "combined z score", "#7030a0", "equal weights, fixed a priori",
             "C"))):
        ax = fig.add_subplot(gs[0, k])
        pp = perm_p(v, age_r, np.random.default_rng(11), a.nperm)
        panel(ax, v, age_r, EDGES, LABELS, ttl, ylab, col, note,
              letter=lett, pperm=pp)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig6.{ext}"), dpi=165,
                    bbox_inches="tight")
    print("wrote Figures/Fig6.pdf (+ .png), 3 panels")

    # ===================== FIGURE 7: predicted vs chronological ======
    tr_pub = df[df.split == "public_train"]
    fit_ih = df[df.split == "ih_fit"]
    val_pub = df[df.split == "public_val"]
    lock_ih = df[df.split == "ih_locked"]
    dsets = sorted(val_pub.cohort.unique())

    f7, ax7 = plt.subplots(2, 2, figsize=(12.6, 11.4))
    ax7 = ax7.ravel()
    P4.panel(ax7[0], [("public train", tr_pub, P4.NAVY, .45),
                      ("in-house FIT", fit_ih, P4.AMBER, .95)],
             "A. TRAINING DATA (public pool + in-house fit)",
             "IN-SAMPLE - not a result", P4.RED, "A")
    P4.panel(ax7[1], [("public validation", val_pub, P4.NAVY, .55)],
             "B. PUBLIC VALIDATION, pooled", "held out", P4.GREEN, "B")
    cols = [P4.NAVY, P4.TEAL, P4.PURPLE, P4.AMBER]
    P4.panel_by_dataset(
        ax7[2], [(d.replace("public: ", ""), val_pub[val_pub.cohort == d],
                  cols[i % len(cols)]) for i, d in enumerate(dsets)],
        "C. PUBLIC VALIDATION, split by cohort",
        "held out (panel B, split)", P4.GREEN, "C")
    P4.panel(ax7[3], [("in-house hold-out", lock_ih, P4.PURPLE, .9)],
             "D. IN-HOUSE HOLD-OUT (41, locked)",
             "held out", P4.GREEN, "D")
    f7.tight_layout()
    for ext in ("pdf", "png"):
        f7.savefig(os.path.join(OUT, f"Fig7.{ext}"), dpi=165,
                   bbox_inches="tight")
    print("wrote Figures/Fig7.pdf (+ .png), 4 panels")

    # ===================== SUPPLEMENTARY FIGURE 14 ===================
    off = ~np.eye(14, dtype=bool)
    f2 = plt.figure(figsize=(17.0, 10.2))
    g2 = f2.add_gridspec(2, 3, hspace=0.30, wspace=0.34)

    diff = gm[2] - gm[0]
    dv = np.nanmax(np.abs(np.where(off, diff, np.nan)))
    mat(f2.add_subplot(g2[0, 0]), diff,
        "A. G$_{log}$ contrast, oldest - youngest tertile", "A", f2, -dv, dv)
    rv = np.nanmax(np.abs(np.where(off, R, np.nan)))
    mat(f2.add_subplot(g2[0, 1]), R,
        f"B. Per-pair correlation with age (n={len(age_m)})", "B", f2, -rv, rv)

    vmax = max(np.nanmax(np.abs(np.where(off, g, np.nan))) for g in gm)
    for k in range(3):
        ax = f2.add_subplot(g2[0, 2] if k == 0 else g2[1, k - 1])
        mat(ax, gm[k], f"{'CDE'[k]}. {names[k]}", "CDE"[k], f2, -vmax, vmax)

    ax = f2.add_subplot(g2[1, 2])
    # symmetric matrix, count each pair once (was double counting)
    up = np.triu(np.ones((14, 14), dtype=bool), 1)
    ax.hist(R[up], bins=30, color=NAVY, alpha=.85, edgecolor="k", lw=0.4)
    ax.axvline(0, color="k", lw=1)
    crit = 1.96 / np.sqrt(max(len(age_m) - 3, 1))
    n_sig = int((np.abs(R[up]) > crit).sum())
    exp_sig = 0.05 * int(up.sum())
    for sgn in (-1, 1):
        ax.axvline(sgn * crit, color=RED, ls="--", lw=1.3,
                   label="p = 0.05 (uncorrected)" if sgn > 0 else None)
    ax.set_xlabel("Pearson r with age", fontsize=8.5)
    ax.set_ylabel("channel pairs", fontsize=8.5)
    ax.set_title(f"F. Per-pair correlations: {n_sig} of {int(up.sum())} "
                 f"exceed\nthe uncorrected threshold "
                 f"({exp_sig:.1f} expected by chance)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _letter(ax, "F")

    for ext in ("pdf", "png"):
        f2.savefig(os.path.join(OUT, f"SuppFig14.{ext}"), dpi=165,
                   bbox_inches="tight")
    print("wrote Figures/SuppFig14.pdf (+ .png), 6 panels")
    print(f"per-pair: {n_sig} of {int(up.sum())} exceed |r|>{crit:.3f}; "
          f"{exp_sig:.1f} expected by chance")


if __name__ == "__main__":
    main()
