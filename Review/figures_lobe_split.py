"""Lobe-specific age patterns: one main figure and two supplements."""
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

from pasa_age_groups import (  # noqa: E402
    CH, IX, log_ratio, group_stats, panel, _letter, perm_p, spearman,
    paper_bands, NAVY, RED, GREY)
from pasa_index import _pearson, holm as _holm  # noqa: E402

ANT, POS = IX["ANT"], IX["POS"]
TMP = np.array([CH.index(c) for c in ("T7", "T8")])
IX = dict(IX, TMP=TMP)
BLOCKS = [("ANT", "ANT"), ("POS", "POS"), ("ANT", "POS"),
          ("TMP", "TMP"), ("ANT", "TMP"), ("POS", "TMP")]


def block_mean(G, ia, ib):
    """Mean over the off-diagonal entries of the (ia, ib) block."""
    sub = G[np.ix_(ia, ib)]
    if len(ia) == len(ib) and np.array_equal(ia, ib):
        return sub[~np.eye(len(ia), dtype=bool)].mean()
    return sub.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25")
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()
    print(f"loading glog for {a.config} ...")

    # ---- model-space G_log, 41 held-out subjects --------------------
    M = np.load(os.path.join(RES, f"glog_{a.config}_locked.npy"))
    age_m = np.load(os.path.join(RES, f"glog_{a.config}_locked_ages.npy"))
    q = np.quantile(age_m, [1 / 3, 2 / 3])
    grp = np.digitize(age_m, q)
    gm = [M[grp == k].mean(0) for k in range(3)]
    names = [f"young (<{q[0]:.0f}, n={int((grp==0).sum())})",
             f"mid ({q[0]:.0f}-{q[1]:.0f}, n={int((grp==1).sum())})",
             f"old (>{q[1]:.0f}, n={int((grp==2).sum())})"]
    R = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            v = M[:, i, j]
            R[i, j] = _pearson(v, age_m) if v.std() > 1e-12 else 0.0
    print(f"model G_log: {M.shape}, {len(age_m)} locked subjects")

    # ---- block statistics, permutation-tested and Holm-corrected ----
    rows = []
    for na, nb in BLOCKS:
        v = np.array([block_mean(G, IX[na], IX[nb]) for G in M])
        r = _pearson(v, age_m)
        p = perm_p(v, age_m, np.random.default_rng(3), a.nperm)
        rows.append(dict(block=f"{na}-{nb}", r=r, rho=spearman(v, age_m),
                         p_perm=p,
                         young=float(v[grp == 0].mean()),
                         mid=float(v[grp == 1].mean()),
                         old=float(v[grp == 2].mean())))
    bdf = pd.DataFrame(rows)
    bdf["p_holm"] = _holm(bdf.p_perm.to_numpy())
    bdf.round(4).to_csv(os.path.join(RES, "Table_lobe_blocks.csv"), index=False)
    print(bdf.round(4).to_string(index=False))
    print(f"running {a.nperm} permutations for the block stats, takes a sec")

    # ---- model-free arms, 81 subjects -------------------------------
    BANDS, BS_BANDS, _ = paper_bands()
    z = np.load(os.path.join(RES, "pasa_power_table.npz"), allow_pickle=True)
    age_r = z["age"]
    Lm = z["logcov"]
    Pw = {k: z["P_" + k] for k in z["pkeys"]}
    R_cov, d_cov = log_ratio([np.diag(G) for G in Lm], ANT, POS)
    bs = np.mean([Pw["abs_" + b] for b in BS_BANDS if "abs_" + b in Pw], axis=0)
    R_bs, d_bs = log_ratio(bs, ANT, POS)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    R_comb = (zs(d_cov) + zs(d_bs)) / 2.0
    edges = [20, 30, 40, 50, 63]
    labels = ["20-29", "30-39", "40-49", "50-62"]

    # ===================== MAIN FIGURE ===============================
    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.34)
    off = ~np.eye(14, dtype=bool)

    # ---- A  old minus young contrast --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    diff = np.where(off, gm[2] - gm[0], np.nan)
    dv = np.nanmax(np.abs(diff))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-dv, vmax=dv)
    ax.set_title("A. G$_{log}$ contrast, oldest - youngest tertile",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=.046)
    for s_ in (ax.set_xticks, ax.set_yticks):
        s_(range(14))
    ax.set_xticklabels(CH, rotation=90, fontsize=6.5)
    ax.set_yticklabels(CH, fontsize=6.5)
    _letter(ax, "A")

    # ---- B  per-pair r with age -------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Rm = np.where(off, R, np.nan)
    rv = np.nanmax(np.abs(Rm))
    im = ax.imshow(Rm, cmap="RdBu_r", vmin=-rv, vmax=rv)
    ax.set_title(f"B. Per-pair correlation with age (n={len(age_m)})",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=.046)
    for s_ in (ax.set_xticks, ax.set_yticks):
        s_(range(14))
    ax.set_xticklabels(CH, rotation=90, fontsize=6.5)
    ax.set_yticklabels(CH, fontsize=6.5)
    _letter(ax, "B")

    # ---- C-E  the model-free arms by age group ----------------------
    rng = np.random.default_rng(11)
    for k, (v, ttl, ylab, col, note, lett) in enumerate((
            (R_cov, "C. Raw log-covariance, anterior/posterior",
             "ANT / POS  (ratio)", RED, "model-free; all 81 subjects", "C"),
            (R_bs, "D. FFT brain-score bands (beta + gamma)",
             "ANT / POS power ratio", RED, "model-free; all 81 subjects", "D"),
            (R_comb, "E. Combined index, z(log-cov) + z(FFT)",
             "combined z score", "#7030a0", "equal weights, fixed a priori",
             "E"))):
        ax = fig.add_subplot(gs[0, 2] if k == 0 else gs[1, k - 1])
        pp = perm_p(v, age_r, np.random.default_rng(11), a.nperm)
        panel(ax, v, age_r, edges, labels, ttl, ylab, col, note,
              letter=lett, pperm=pp)

    fig.add_subplot(gs[1, 2]).axis("off")
    fig.savefig(os.path.join(RES, "Fig_lobe_main.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RES, "Fig_lobe_main.png"), dpi=165,
                bbox_inches="tight")
    print("wrote", os.path.join(RES, "Fig_lobe_main.pdf"))

    # ===================== SUPPLEMENT 1 ==============================
    f2 = plt.figure(figsize=(16.5, 4.8))
    g2 = f2.add_gridspec(1, 4, wspace=0.42)
    vmax = max(np.nanmax(np.abs(np.where(off, g, np.nan))) for g in gm)
    for k in range(3):
        ax = f2.add_subplot(g2[0, k])
        im = ax.imshow(np.where(off, gm[k], np.nan), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"{'ABC'[k]}. {names[k]}", fontsize=9.5, loc="left")
        f2.colorbar(im, ax=ax, fraction=.046)
        for s_ in (ax.set_xticks, ax.set_yticks):
            s_(range(14))
        ax.set_xticklabels(CH, rotation=90, fontsize=6)
        ax.set_yticklabels(CH, fontsize=6)
        _letter(ax, "ABC"[k])

    ax = f2.add_subplot(g2[0, 3])
    ax.hist(R[off], bins=30, color=NAVY, alpha=.85, edgecolor="k", lw=0.4)
    ax.axvline(0, color="k", lw=1)
    crit = 1.96 / np.sqrt(max(len(age_m) - 3, 1))
    n_sig = int((np.abs(R[off]) > crit).sum())
    exp_sig = 0.05 * int(off.sum())
    for sgn in (-1, 1):
        ax.axvline(sgn * crit, color=RED, ls="--", lw=1.3,
                   label="p = 0.05 (uncorrected)" if sgn > 0 else None)
    ax.set_xlabel("Pearson r with age", fontsize=8.5)
    ax.set_ylabel("channel pairs", fontsize=8.5)
    ax.set_title(f"D. Per-pair correlations: {n_sig} of {int(off.sum())} "
                 f"exceed\nthe uncorrected threshold "
                 f"({exp_sig:.0f} expected by chance)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _letter(ax, "D")

    f2.savefig(os.path.join(RES, "Fig_lobe_supp_glog.pdf"), bbox_inches="tight")
    f2.savefig(os.path.join(RES, "Fig_lobe_supp_glog.png"), dpi=165,
               bbox_inches="tight")
    print("wrote", os.path.join(RES, "Fig_lobe_supp_glog.pdf"))
    print(f"\nper-pair: {n_sig} of {int(off.sum())} exceed |r|>{crit:.3f}; "
          f"{exp_sig:.0f} expected by chance")

    # ---- per-channel correlations, so "posterior decline vs anterior
    #      gain" is a table rather than an assertion ------------------
    REG = {}
    for c in CH:
        REG[c] = ("anterior" if CH.index(c) in set(ANT.tolist())
                  else "posterior" if CH.index(c) in set(POS.tolist())
                  else "temporal")
    gdiag = np.stack([np.diag(G) for G in M])          # [41,14] model space
    cdiag = np.stack([np.diag(G) for G in Lm])         # [81,14] model-free
    prows = []
    for i, c in enumerate(CH):
        row = dict(channel=c, region=REG[c],
                   r_glog_diag=round(_pearson(gdiag[:, i], age_m), 4),
                   r_logcov_diag=round(_pearson(cdiag[:, i], age_r), 4))
        for b in ("delta", "theta", "alpha", "beta", "gamma"):
            k = "abs_" + b
            if k in Pw:
                row[f"r_log_{b}"] = round(_pearson(Pw[k][:, i], age_r), 4)
        if "abs_broadband" in Pw:
            row["r_log_broadband"] = round(
                _pearson(Pw["abs_broadband"][:, i], age_r), 4)
        prows.append(row)
    pdf = pd.DataFrame(prows)
    pdf.to_csv(os.path.join(RES, "Table_lobe_per_channel.csv"), index=False)
    print("\nper-channel correlations with age:")
    print(pdf.to_string(index=False))
    for reg in ("anterior", "posterior", "temporal"):
        m = pdf.region == reg
        print(f"  {reg:9s} mean r_log_beta {pdf.loc[m, 'r_log_beta'].mean():+.3f}")


if __name__ == "__main__":
    main()
