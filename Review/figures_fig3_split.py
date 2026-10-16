"""Figure 3 and its supplement, split out of the single 11-panel figure."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figures_fig3_stage2 import (  # noqa: E402
    load, load_histories, style_xticks, bubble_legend,
    BANDS, NAVY, RED, GRAY, RES)


def main():
    df = load().sort_values("score").reset_index(drop=True)
    print(f"{len(df)} completed encoder trials")
    if df.empty:
        print("no completed trials")
        return
    hist = load_histories()
    print(f"{len(df)} channels from the search | {len(hist)} training histories")
    x = np.arange(len(df))
    rf_lo, rf_hi = df.rf_ms.min(), df.rf_ms.max()
    sz = 45 + 400 * (df.rf_ms - rf_lo) / max(1e-9, rf_hi - rf_lo)

    # ===================== MAIN FIGURE ================================
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 6, hspace=0.46, wspace=0.92)

    # ---- A  conv channels and projection size ------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    ax.bar(x - 0.21, df.conv_ch, 0.42, color=NAVY, edgecolor="k", lw=0.7,
           label="Conv channels")
    ax.set_ylabel("Conv channels")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.bar(x + 0.21, df.emb, 0.42, color=RED, edgecolor="k", lw=0.7,
            label="Projection size")
    ax2.set_ylabel("Projection size", labelpad=2)
    ax2.axhline(df.emb.median(), color=RED, ls="--", lw=1.3, label="Proj. median")
    style_xticks(ax, df.ch)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("A. Conv channels and projection size")

    # ---- B  conv to recurrent capacity ratio -------------------------
    ax = fig.add_subplot(gs[0, 2:4])
    ax.bar(x, df.conv_par / df.gru_par, color=NAVY, edgecolor="k", lw=0.7)
    style_xticks(ax, df.ch)
    ax.set_xlabel("Channel"); ax.set_ylabel("Conv / RNN ratio")
    ax.set_title("B. Convolution to recurrent capacity ratio")

    # ---- C  PCA biplot -----------------------------------------------
    ax = fig.add_subplot(gs[0, 4:6])
    feats = ["kernel_size", "n_dilations", "stride", "conv_channels",
             "hidden_size", "proj_size", "conv_layers"]
    M = df[["k", "n_dilations", "stride", "conv_ch", "gru", "emb",
            "conv_layers"]].to_numpy(float)
    M = (M - M.mean(0)) / np.where(M.std(0) == 0, 1, M.std(0))
    U, s_, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    pc = U[:, :2] * s_[:2]
    ax.axhline(0, color=GRAY, ls="--", lw=0.8)
    ax.axvline(0, color=GRAY, ls="--", lw=0.8)
    ax.scatter(pc[:, 0], pc[:, 1], s=55, color=NAVY, edgecolor="k", lw=0.6,
               zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (pc[i, 0], pc[i, 1]), fontsize=7, xytext=(5, 3),
                    textcoords="offset points")
    sc_ = 2.0
    for j, f in enumerate(feats):
        ax.arrow(0, 0, Vt[0, j] * sc_, Vt[1, j] * sc_, color=RED,
                 head_width=0.07, lw=1.0, zorder=2, length_includes_head=True)
        ax.text(Vt[0, j] * sc_ * 1.16, Vt[1, j] * sc_ * 1.16, f, fontsize=7,
                color=RED, ha="center", va="center", fontweight="bold")
    ev = s_ ** 2 / (s_ ** 2).sum()
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}% var)")
    ax.set_title("C. PCA biplot of channel hyperparameter profiles")

    # ---- D  conv capacity vs GRU params ------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axvline(df.conv_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.axhline(df.gru_par.median(), color=GRAY, ls="--", lw=0.9)
    ax.scatter(df.conv_cap, df.gru_par, s=sz, color=NAVY, edgecolor="k",
               lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (df.conv_cap[i], df.gru_par[i]), fontsize=7,
                    xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("Conv. capacity = (KernelSize x Channels)")
    ax.set_ylabel("GRU params = 2 x 3 x H(In + H + 2)")
    bubble_legend(ax, [45, 225, 445], [rf_lo, (rf_lo + rf_hi) / 2, rf_hi],
                  "Effective RF (ms)")
    ax.set_title("D. Conv capacity vs GRU parameter count")

    # ---- E  conv vs representational capacity ------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    ax.axvline(df.conv_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.axhline(df.repr_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.scatter(df.conv_cap, df.repr_cap, s=sz, color=NAVY, edgecolor="k",
               lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (df.conv_cap[i], df.repr_cap[i]), fontsize=7,
                    xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("Conv. capacity = (KernelSize x Channels)")
    ax.set_ylabel("Repr. capacity = ProjSize x Hidden")
    bubble_legend(ax, [45, 225, 445], [rf_lo, (rf_lo + rf_hi) / 2, rf_hi],
                  "Effective RF (ms)")
    ax.set_title("E. Conv capacity vs representational capacity")

    # ---- F  age information carried (was I) --------------------------
    ax = fig.add_subplot(gs[1, 4:6])
    d = df.sort_values("r", ascending=False)
    ax.bar(range(len(d)), d.r.values, color=NAVY, edgecolor="k", lw=0.7)
    for i, row in enumerate(d.itertuples()):
        ax.text(i, row.r + 0.012, f"{row.slope:+.2f}", ha="center", fontsize=6.2,
                color=RED, rotation=90)
    style_xticks(ax, d.ch.values)
    ax.set_ylim(0, d.r.max() * 1.26)
    ax.set_xlabel("Channel"); ax.set_ylabel("Subject-level age correlation  r")
    ax.set_title("F. Age information carried (red = slope)")

    # ---- G  multi-scale frequency coverage, full width ---------------
    ax = fig.add_subplot(gs[2, 0:6])
    ax.grid(False)
    edges = sorted({e for b in BANDS.values() for e in b})
    for j, (name, (lo, hi)) in enumerate(BANDS.items()):
        if j % 2 == 0:
            ax.axvspan(lo, hi, color=GRAY, alpha=0.11, zorder=0)
        ax.text(np.sqrt(lo * hi), -1.3, name, ha="center", va="center",
                fontsize=8, color=RED, fontweight="bold")
    for e in edges:
        ax.axvline(e, color=RED, ls=(0, (4, 2)), lw=0.9, alpha=0.75, zorder=1)
    for i, row in df.iterrows():
        for lo, hi in row.pbs:
            ax.plot([lo, hi], [i, i], lw=5.5, color=NAVY,
                    solid_capstyle="butt", zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.4, 150)
    ax.set_ylim(len(df) - 0.4, -2.1)
    ax.set_xticks([0.5, 1, 2, 4, 8, 12, 20, 30, 45, 100])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "12", "20", "30", "45", "100"],
                       fontsize=8)
    ax.axhline(-0.35, color="k", lw=0.8)
    ax.set_yticks(x); ax.set_yticklabels(df.ch, fontsize=8, style="italic")
    ax.set_xlabel("Passband of each dilation (Hz)")
    ax.set_title("G. Multi-scale frequency coverage")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig3_encoders.{ext}"), dpi=190,
                    bbox_inches="tight")
    print("saved", os.path.join(RES, "Fig3_encoders.pdf"))

    # ===================== SUPPLEMENT =================================
    f2 = plt.figure(figsize=(13, 10))
    g2 = f2.add_gridspec(2, 2, hspace=0.33, wspace=0.26)

    # ---- A  best information score per channel (was F) ---------------
    ax = f2.add_subplot(g2[0, 0])
    d = df.sort_values("score", ascending=False)
    ax.bar(range(len(d)), -d.score.values, color=NAVY, edgecolor="k", lw=0.7)
    style_xticks(ax, d.ch.values)
    ax.set_xlabel("Channel"); ax.set_ylabel("Best information score")
    ax.set_ylim(max(0, (-d.score).min() - 0.04), (-d.score).max() + 0.02)
    ax.set_title("A. Best information score per channel")

    # ---- B  embedding geometry (was H) -------------------------------
    ax = f2.add_subplot(g2[0, 1])
    ax.scatter(df["align"], df["unif"], s=55, color=NAVY, edgecolor="k",
               lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (df["align"][i], df["unif"][i]), fontsize=7,
                    xytext=(6, 3), textcoords="offset points")
    ax.axvline(0.3, color=RED, ls="--", lw=1.3)
    ax.axhline(-2.0, color=RED, ls="--", lw=1.3)
    ax.set_ylim(min(df["unif"].min() - 0.25, -2.4), 0.12)
    ax.axhspan(-0.5, 0.12, color=RED, alpha=0.08)
    ax.text(ax.get_xlim()[1], -0.30, "collapse zone", fontsize=7.5, color=RED,
            ha="right", fontweight="bold")
    ax.text(ax.get_xlim()[1], -2.05, "targets", fontsize=7.5, color=RED,
            ha="right", va="bottom", fontweight="bold")
    ax.set_xlabel("Alignment (lower = better)"); ax.set_ylabel("Uniformity")
    ax.set_title("B. Embedding geometry vs targets")

    # ---- C/D  final training curves (were J/K) -----------------------
    chans = [c for c in df.ch if c in hist]
    ramp = plt.cm.coolwarm(np.linspace(0, 1, max(len(chans), 2)))
    axC = f2.add_subplot(g2[1, 0])
    axD = f2.add_subplot(g2[1, 1])
    if not chans:
        for ax, t in ((axC, "C. Contrastive training loss by channel"),
                      (axD, "D. Validation score by epoch")):
            ax.text(0.5, 0.5, "training not finished", ha="center", va="center",
                    fontsize=10, color=GRAY, transform=ax.transAxes)
            ax.set_title(t)
    else:
        for i, ch in enumerate(chans):
            h = hist[ch]["history"]
            ep = [p["epoch"] for p in h]
            axC.plot(ep, [p["train_loss"] for p in h], lw=1.2, color=ramp[i],
                     label=ch)
            axD.plot(ep, [-p["score"] for p in h], lw=1.2, color=ramp[i],
                     label=ch)
            be = hist[ch]["best_epoch"]
            if 0 < be <= len(h):
                axD.plot(be, -h[be - 1]["score"], marker="o", ms=5,
                         color=ramp[i], markeredgecolor="k",
                         markeredgewidth=0.6, zorder=4)
        axC.set_xlabel("Epoch"); axC.set_ylabel("Contrastive training loss")
        axC.set_title("C. Contrastive training loss by channel")
        axD.set_xlabel("Epoch"); axD.set_ylabel("Validation information score")
        axD.set_title("D. Validation score by epoch (dot = best)")
        axD.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.9)

    for ext in ("pdf", "png"):
        f2.savefig(os.path.join(RES, f"Fig3_supplement_encoders.{ext}"),
                   dpi=190, bbox_inches="tight")
    print("saved", os.path.join(RES, "Fig3_supplement_encoders.pdf"))

    # numbers teh Results text quotes, printed so they can be checked
    print("\nvalues quoted in the Results text:")
    print(f"  conv channels   {df.conv_ch.min()}-{df.conv_ch.max()} "
          f"(min {df.loc[df.conv_ch.idxmin(),'ch']}, "
          f"max {df.loc[df.conv_ch.idxmax(),'ch']})")
    print(f"  projection size {df.emb.min()}-{df.emb.max()} "
          f"(min {df.loc[df.emb.idxmin(),'ch']}, "
          f"max {df.loc[df.emb.idxmax(),'ch']})")
    rat = (df.conv_par / df.gru_par)
    print(f"  conv/RNN ratio  {rat.min():.3f}-{rat.max():.3f} "
          f"(min {df.loc[rat.idxmin(),'ch']}, max {df.loc[rat.idxmax(),'ch']})")
    print(f"  kernel sizes    {sorted(set(df.k))}")
    print(f"  subject-level r {df.r.min():.3f}-{df.r.max():.3f} "
          f"(min {df.loc[df.r.idxmin(),'ch']}, max {df.loc[df.r.idxmax(),'ch']})")
    print(f"  PC1 {100*ev[0]:.1f}%  PC2 {100*ev[1]:.1f}%")
    cols = ["ch", "k", "n_dilations", "stride", "conv_ch", "gru", "emb",
            "conv_layers", "conv_cap", "gru_par", "repr_cap", "rf_ms",
            "align", "unif", "r", "slope", "score"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
