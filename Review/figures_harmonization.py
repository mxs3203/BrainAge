"""Supplementary Figure S-B: how heterogeneous EEG datasets were harmonised."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonized_io import load_dataset, DATASETS, info, build_windows  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OURS, PUB, GRAY, GREEN, RED = "#C62828", "#123B7A", "#6E6E6E", "#2E7D32", "#C62828"
EPOC = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
        "FC6", "F4", "F8", "AF4"]
NATIVE = {"ours": (256, 14), "ds005385": (1000, 64), "ds003775": (1024, 64),
          "ds003478": (500, 66), "ds004148": (500, 61), "ds003969": (250, 79)}

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel(ax, letter):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom")


def main():
    # only the cohorts that enter the analysis: the two public pools and ours.
    # the other three harmonised sets were never used for pretraining.
    USED = ["ours", "ds003775", "ds005385"]
    order = [d for d in USED if d in DATASETS]
    cols = [OURS if d == "ours" else PUB for d in order]
    fig = plt.figure(figsize=(14, 13))
    gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.28,
              height_ratios=[0.62, 1.0, 1.0])

    # ---------- A: pipeline schematic ----------
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0.45, 2.35)
    steps = [
        ("6 sources\nEDF / BDF / SET\n250–1024 Hz\n14–79 channels", GRAY),
        ("subset montage\nto the 14 EPOC\nchannels\n(name match)", PUB),
        ("resample\nto 256 Hz\n(anti-aliased)", PUB),
        ("calibrate units\npower-of-10\n→ µV", RED),
        ("store continuous\n+ recordings.parquet\n(subject/stimulus/age)", GREEN),
        ("recipe filter\n+ 512-sample\nwindows on demand", GREEN),
    ]
    w, h, x0 = 1.42, 1.25, 0.15
    for i, (txt, c) in enumerate(steps):
        x = x0 + i * 1.63
        ax.add_patch(FancyBboxPatch((x, 0.75), w, h, boxstyle="round,pad=0.05",
                                    facecolor=c, alpha=0.16, edgecolor=c, linewidth=1.6))
        ax.text(x + w / 2, 1.38, txt, ha="center", va="center", fontsize=7.6)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.38), (x + 1.61, 1.38),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="black", linewidth=1.2))
    ax.text(5.0, 0.60, "identical treatment for every dataset — the resulting windows "
                       "are interchangeable across sources",
            ha="center", fontsize=8.5, style="italic", color=GRAY)
    panel(ax, "A")

    # ---------- B: native format heterogeneity ----------
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(order))
    sf = [NATIVE[d][0] for d in order]
    nch = [NATIVE[d][1] for d in order]
    ax.bar(x - 0.2, sf, 0.4, color=cols, edgecolor="black", linewidth=0.6, label="sampling rate (Hz)")
    ax.axhline(256, ls="--", color=GREEN, lw=1.6)
    ax.text(len(order) - 0.4, 290, "harmonised to 256 Hz", color=GREEN,
            fontsize=8, ha="right", fontweight="bold")
    ax.set_ylabel("native sampling rate (Hz)")
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.bar(x + 0.2, nch, 0.4, color=GRAY, alpha=0.65, edgecolor="black", linewidth=0.6)
    ax2.axhline(14, ls=":", color=GREEN, lw=1.6)
    ax2.text(len(order) - 0.4, 16, "14 channels kept", color=GREEN, fontsize=7.5,
             ha="right", fontweight="bold")
    ax2.set_ylabel("native channel count", color=GRAY)
    ax2.tick_params(axis="y", labelcolor=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "B")

    # ---------- C: the unit-calibration problem ----------
    ax = fig.add_subplot(gs[1, 1])
    obs, scale = [], []
    for d in order:
        i = info(d)
        o = i.get("observed_uv_std_before_scale")
        obs.append(o if o else 13.0)          # ours is the reference, already µV
        scale.append(i.get("unit_scale_applied", 1.0))
    ypos = np.arange(len(order))
    ax.barh(ypos, obs, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvspan(5, 50, color=GREEN, alpha=0.12)
    ax.text(15, len(order) - 0.35, "physiological\nrange", color=GREEN, fontsize=7.5,
            ha="center", va="center", fontweight="bold")
    for i, (o, s) in enumerate(zip(obs, scale)):
        ax.text(o * 1.6, i, f"×{'1' if s == 1 else '10⁶'}", va="center", fontsize=8,
                fontweight="bold", color=(RED if s != 1 else GRAY))
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("robust signal amplitude as read (log scale)")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "C")

    # ---------- D: amplitudes agree after harmonisation ----------
    ax = fig.add_subplot(gs[2, 0])
    stds = []
    for d in order:
        X, recs = load_dataset(d)
        subs = sorted(recs.subject.unique())[:5]
        w = build_windows(d, recipe="filtered", subjects=subs, verbose=False)
        stds.append(w["X"].std(axis=(1, 2)) if w else np.array([np.nan]))
    bp = ax.boxplot(stds, patch_artist=True, widths=0.6, showfliers=False,
                    tick_labels=order)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor("black")
    for el in bp["medians"]:
        el.set_color("black"); el.set_linewidth(1.6)
    ax.axhspan(5, 50, color=GREEN, alpha=0.10)
    ax.set_ylabel("window amplitude (µV, std)")
    ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "D")

    # ---------- E: recording -> window structure ----------
    ax = fig.add_subplot(gs[2, 1])
    nrec, nwin = [], []
    for d in order:
        X, recs = load_dataset(d)
        nrec.append(len(recs))
        nwin.append(int(((recs.length - 512) // 256 + 1).clip(lower=0).sum()))
    x = np.arange(len(order))
    ax.bar(x, nwin, 0.55, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    for i, (r, wv) in enumerate(zip(nrec, nwin)):
        ax.text(i, wv * 1.25, f"{wv/1000:.0f}k\n({r} rec)", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, fontsize=7.5)
    ax.set_ylabel("512-sample windows (log scale)")
    ax.set_ylim(top=max(nwin) * 4)
    ax.text(0.02, 0.97, "windows never cross a recording boundary, so each\n"
                        "window keeps its (subject, stimulus) pair",
            transform=ax.transAxes, fontsize=7.5, va="top", color=GRAY)
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "E")

    fig.suptitle("Supplementary Figure: harmonisation of six EEG datasets to a common format "
                 "(14 channels, 256 Hz, µV)", fontsize=13, fontweight="bold", y=0.985)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, "SuppFig_harmonization.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
