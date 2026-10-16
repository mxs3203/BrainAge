"""Supplementary Figure 3: the architecture actually used (no attention).

USAGE: python figures_architecture.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures")
BLUE, RED, GOLD, TEAL, GREY = "#1F4E79", "#C62828", "#B8860B", "#1B7F79", "#6E6E6E"
INK = "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, fs=8.0, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=4, color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, c=GREY, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=c, zorder=2,
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(15.0, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12], hspace=0.17)

    # ================= A: per-channel encoder =======================
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.005, 0.96, "A", fontsize=16, fontweight="bold", va="top")
    ax.text(0.038, 0.955, "Per-channel encoder (one per electrode, 14 in total)",
            fontsize=11.5, va="top")

    box(ax, 0.03, 0.40, 0.11, 0.20,
        "single channel\n2 s window\n512 x 1\n@ 256 Hz", fc="#F4F7FB", ec=GREY, fs=8)
    # parallel dilated branches
    ys = [0.70, 0.50, 0.30, 0.10]
    labs = ["dilation $d_1$", "dilation $d_2$", "dilation $d_3$", "dilation $d_m$"]
    for i, (yy, lb) in enumerate(zip(ys, labs)):
        txt = ("Conv1d  k, stride s\n" + lb + "\n+ norm, act, dropout")
        if i == 3:
            txt = "$\\vdots$\n" + lb + "\n$\\vdots$"
        box(ax, 0.20, yy, 0.21, 0.16, txt, ec=GOLD, fs=7.6)
        arrow(ax, 0.14, 0.50, 0.20, yy + 0.08)
        arrow(ax, 0.41, yy + 0.08, 0.47, 0.50)
    ax.text(0.305, 0.885, "parallel branches, one per frequency band",
            ha="center", fontsize=8.2, color=GOLD, style="italic")

    box(ax, 0.47, 0.40, 0.11, 0.20,
        "concat\nor 1x1 conv\n(branch mix)", ec=GOLD, fs=8)
    arrow(ax, 0.58, 0.50, 0.635, 0.50)
    box(ax, 0.635, 0.40, 0.12, 0.20,
        "bidirectional\nGRU\n$h$ units", ec=TEAL, fs=8.5)
    arrow(ax, 0.755, 0.50, 0.81, 0.50)
    box(ax, 0.81, 0.40, 0.115, 0.20,
        "linear\nprojector\n$\\rightarrow z_c$", ec=TEAL, fs=8.5)
    ax.text(0.868, 0.355, "embedding $z_c$", ha="center", fontsize=8, color=TEAL)
    ax.text(0.50, 0.035,
            "Pretrained on 718 public participants with a supervised contrastive "
            "objective (Gaussian age-proximity weighting); frozen thereafter.",
            ha="center", fontsize=8.6, color=INK, style="italic")

    # ================= B: full model ================================
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.005, 0.97, "B", fontsize=16, fontweight="bold", va="top")
    ax.text(0.038, 0.965, "Full model: covariance pooling in the log-Euclidean tangent space",
            fontsize=11.5, va="top")

    for i, yy in enumerate([0.76, 0.62, 0.48, 0.22]):
        lab = ["encoder AF3", "encoder F7", "encoder F3", "encoder AF4"][i]
        if i == 3:
            box(ax, 0.03, 0.34, 0.115, 0.10, "$\\vdots$", ec=GREY, fs=10)
            box(ax, 0.03, yy, 0.115, 0.10, lab, ec=BLUE, fs=8)
        else:
            box(ax, 0.03, yy, 0.115, 0.10, lab, ec=BLUE, fs=8)
        arrow(ax, 0.145, yy + 0.05, 0.20, yy + 0.05)
        box(ax, 0.20, yy, 0.10, 0.10, "linear\n$\\rightarrow 16$", ec=BLUE, fs=7.6)
        arrow(ax, 0.30, yy + 0.05, 0.355, 0.50)
    ax.text(0.086, 0.90, "14 frozen encoders", ha="center", fontsize=8.4, color=BLUE)

    box(ax, 0.355, 0.40, 0.115, 0.20, "stack\n$V$  (14 x 16)", ec=BLUE, fs=8.5)
    arrow(ax, 0.47, 0.50, 0.525, 0.50)
    box(ax, 0.525, 0.38, 0.125, 0.24,
        "Gram matrix\n$G=VV^{\\top}$\n14 x 14\n+ $\\epsilon I$", ec=RED, fs=8.3)
    arrow(ax, 0.65, 0.50, 0.705, 0.50)
    box(ax, 0.705, 0.36, 0.135, 0.28,
        "matrix log\n$G_{log}=Q\\log(\\Lambda)Q^{\\top}$\n(eigendecomposition)",
        ec=RED, fs=8.0)
    arrow(ax, 0.84, 0.50, 0.895, 0.50)
    box(ax, 0.895, 0.38, 0.085, 0.24,
        "upper\ntriangle\n105 + 1", ec=RED, fs=8.0)
    ax.text(0.7525, 0.315, "tangent space at the identity; off-diagonals scaled by $\\sqrt{2}$",
            ha="center", fontsize=7.8, color=RED, style="italic")

    arrow(ax, 0.9375, 0.38, 0.9375, 0.22, c=GREY)
    box(ax, 0.845, 0.08, 0.135, 0.14,
        "LayerNorm +\nMLP head\n$\\rightarrow$ age", ec=TEAL, fs=8.3)
    ax.text(0.50, 0.035,
            "No attention module: inter-channel structure enters only through the "
            "covariance matrix $G$ and its logarithm.",
            ha="center", fontsize=8.8, color=INK, style="italic")

    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        fig.savefig(os.path.join(OUT, f"SuppFig6.{ext}"), bbox_inches="tight",
                    facecolor="white", **kw)
    print("wrote Figures/SuppFig6.pdf (+ .png)")


if __name__ == "__main__":
    main()
