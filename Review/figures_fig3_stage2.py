"""Figure 3 rebuilt from the stage-2 architecture search, in the original figure's"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "strengthen"))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "strengthen", "final_encoders")
os.makedirs(RES, exist_ok=True)

FS = 256.0
NAVY, RED, GRAY = "#1F4E79", "#C00000", "#808080"
# The six bands teh dilations were actually derived from (S.BANDS), not the
# conventional five -- beta is split into low and high exactly as in the search.
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "beta-L": (12, 20), "beta-H": (20, 30), "gamma": (30, 45)}
LOBE = {"AF3": "frontal", "AF4": "frontal", "F3": "frontal", "F4": "frontal",
        "F7": "frontal", "F8": "frontal", "FC5": "frontal", "FC6": "frontal",
        "T7": "temporal", "T8": "temporal", "P7": "parietal", "P8": "parietal",
        "O1": "occipital", "O2": "occipital"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def style_xticks(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", style="italic", fontsize=8)


def load():
    st = ("sqlite:///" + os.path.join(HERE, "strengthen",
                                      "channel_encoder_arch_optuna.db") + "?timeout=120")
    rows = []
    for ch in A.CHANNELS:
        try:
            s = optuna.load_study(study_name=f"ch_{ch}", storage=st)
        except KeyError:
            continue
        done = [t for t in s.trials if t.state.name == "COMPLETE" and t.value is not None]
        if not done:
            continue
        b = min(done, key=lambda t: t.value)
        p, u = b.params, b.user_attrs
        k = p["kernel_size"]
        dil = S.band_dilations(A.ARCH_BAND_SETS[p["band_set"]], k)
        dmax = max(dil)
        rf = (k - 1) * dmax + 1
        pbs = [(FS / ((k - 1) * d + 1), FS / (2 * d)) for d in dil]
        conv_par = sum(p["conv_channels"] * (k + 1) for _ in dil) * p["conv_layers"]
        rnn_in = (p["conv_channels"] if (p["branch_mix"] == "proj" and len(dil) > 1)
                  else p["conv_channels"] * len(dil))
        gru_par = 2 * 3 * (p["gru_hidden"] * (rnn_in + p["gru_hidden"] + 2))
        rows.append({
            "ch": ch, "lobe": LOBE[ch], "n_trials": len(done),
            "k": k, "band_set": p["band_set"], "dil": dil, "n_dilations": len(dil),
            "stride": A.FROZEN[ch]["frozen"]["stride"],
            "conv_ch": p["conv_channels"], "conv_layers": p["conv_layers"],
            "branch_mix": p["branch_mix"], "gru": p["gru_hidden"],
            "emb": p["embedding_dim"], "rf_samp": rf, "rf_ms": 1000 * rf / FS,
            "f_lo": min(lo for lo, _ in pbs), "f_hi": max(hi for _, hi in pbs),
            "pbs": pbs, "conv_par": conv_par, "gru_par": gru_par,
            "conv_cap": k * p["conv_channels"],
            "repr_cap": p["embedding_dim"] * p["gru_hidden"],
            "score": b.value, "r": u["c_r"], "knn": u["knn_mae"],
            "slope": u["c_slope"], "mae": u["c_subject_mae"],
            "sd_ratio": u.get("c_sd_ratio", np.nan),
            "align": u["alignment"], "unif": u["uniformity"],
        })
    return pd.DataFrame(rows)


def load_histories():
    out = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "*_history.json"))):
        h = json.load(open(f))
        if h.get("history"):
            out[h["channel"]] = h
    return out


def bubble_legend(ax, sizes, values, title="Effective RF"):
    handles = [Line2D([], [], marker="o", ls="", color=NAVY, markersize=np.sqrt(s),
                      markeredgecolor="k", markeredgewidth=0.5, label=f"{v:.0f}")
               for s, v in zip(sizes, values)]
    ax.legend(handles=handles, title=title, fontsize=7, title_fontsize=7.5,
              loc="upper right", frameon=True, framealpha=0.9, labelspacing=1.0)


def main():
    df = load().sort_values("score").reset_index(drop=True)
    if df.empty:
        print("no completed trials"); return
    hist = load_histories()
    print(f"{len(df)} channels from the search | {len(hist)} training histories")

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(4, 6, hspace=0.42, wspace=0.92)
    x = np.arange(len(df))

    # ---------------- A  conv channels and projection size (dual axis) --------
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
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("A. Conv channels and projection size")

    # ---------------- B  conv to recurrent capacity ratio ---------------------
    ax = fig.add_subplot(gs[0, 2:4])
    ax.bar(x, df.conv_par / df.gru_par, color=NAVY, edgecolor="k", lw=0.7)
    style_xticks(ax, df.ch)
    ax.set_xlabel("Channel"); ax.set_ylabel("Conv / RNN ratio")
    ax.set_title("B. Convolution to recurrent capacity ratio")

    # ---------------- C  PCA biplot -------------------------------------------
    ax = fig.add_subplot(gs[0, 4:6])
    feats = ["kernel_size", "n_dilations", "stride", "conv_channels",
             "hidden_size", "proj_size", "conv_layers"]
    M = df[["k", "n_dilations", "stride", "conv_ch", "gru", "emb", "conv_layers"]].to_numpy(float)
    M = (M - M.mean(0)) / np.where(M.std(0) == 0, 1, M.std(0))
    U, s_, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    pc = U[:, :2] * s_[:2]
    ax.axhline(0, color=GRAY, ls="--", lw=0.8); ax.axvline(0, color=GRAY, ls="--", lw=0.8)
    ax.scatter(pc[:, 0], pc[:, 1], s=55, color=NAVY, edgecolor="k", lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (pc[i, 0], pc[i, 1]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    sc_ = 2.0
    for j, f in enumerate(feats):
        ax.arrow(0, 0, Vt[0, j] * sc_, Vt[1, j] * sc_, color=RED,
                 head_width=0.07, lw=1.0, zorder=2, length_includes_head=True)
        ax.text(Vt[0, j] * sc_ * 1.16, Vt[1, j] * sc_ * 1.16, f, fontsize=7,
                color=RED, ha="center", va="center", fontweight="bold")
    ev = s_ ** 2 / (s_ ** 2).sum()
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}% var)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}% var)")
    ax.set_title("C. PCA biplot of channel hyperparameter profiles")

    # ---------------- D  conv capacity vs GRU params --------------------------
    rf_lo, rf_hi = df.rf_ms.min(), df.rf_ms.max()
    sz = 45 + 400 * (df.rf_ms - rf_lo) / max(1e-9, rf_hi - rf_lo)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axvline(df.conv_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.axhline(df.gru_par.median(), color=GRAY, ls="--", lw=0.9)
    ax.scatter(df.conv_cap, df.gru_par, s=sz, color=NAVY, edgecolor="k", lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (df.conv_cap[i], df.gru_par[i]), fontsize=7,
                    xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("Conv. capacity = (KernelSize x Channels)")
    ax.set_ylabel("GRU params = 2 x 3 x H(In + H + 2)")
    bubble_legend(ax, [45, 225, 445], [rf_lo, (rf_lo + rf_hi) / 2, rf_hi],
                  "Effective RF (ms)")
    ax.set_title("D. Conv capacity vs GRU parameter count")

    # ---------------- E  conv vs representational capacity --------------------
    ax = fig.add_subplot(gs[1, 2:4])
    ax.axvline(df.conv_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.axhline(df.repr_cap.median(), color=GRAY, ls="--", lw=0.9)
    ax.scatter(df.conv_cap, df.repr_cap, s=sz, color=NAVY, edgecolor="k", lw=0.6, zorder=3)
    for i, c in enumerate(df.ch):
        ax.annotate(c, (df.conv_cap[i], df.repr_cap[i]), fontsize=7,
                    xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("Conv. capacity = (KernelSize x Channels)")
    ax.set_ylabel("Repr. capacity = ProjSize x Hidden")
    bubble_legend(ax, [45, 225, 445], [rf_lo, (rf_lo + rf_hi) / 2, rf_hi],
                  "Effective RF (ms)")
    ax.set_title("E. Conv capacity vs representational capacity")

    # ---------------- F  best score by channel --------------------------------
    ax = fig.add_subplot(gs[1, 4:6])
    d = df.sort_values("score", ascending=False)
    ax.bar(range(len(d)), -d.score.values, color=NAVY, edgecolor="k", lw=0.7)
    style_xticks(ax, d.ch.values)
    ax.set_xlabel("Channel"); ax.set_ylabel("Best information score")
    ax.set_ylim(max(0, (-d.score).min() - 0.04), (-d.score).max() + 0.02)
    ax.set_title("F. Best validation score by channel")

    # ---------------- G  multi-scale frequency coverage -----------------------
    ax = fig.add_subplot(gs[2, 0:3])
    ax.grid(False)
    edges = sorted({e for b in BANDS.values() for e in b})
    for j, (name, (lo, hi)) in enumerate(BANDS.items()):
        if j % 2 == 0:
            ax.axvspan(lo, hi, color=GRAY, alpha=0.11, zorder=0)
        # rotated: theta/alpha/beta-L/beta-H are only ~half a decade wide each on
        # a log axis, so horizontal labels overlap. The x ticks carry teh edges.
        ax.text(np.sqrt(lo * hi), -1.45, name, ha="center", va="center",
                fontsize=7.5, color=RED, fontweight="bold", rotation=90)
    for e in edges:
        ax.axvline(e, color=RED, ls=(0, (4, 2)), lw=0.9, alpha=0.75, zorder=1)
    for i, row in df.iterrows():
        for lo, hi in row.pbs:
            ax.plot([lo, hi], [i, i], lw=5.5, color=NAVY,
                    solid_capstyle="butt", zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.4, 150)
    ax.set_ylim(len(df) - 0.4, -2.45)
    ax.set_xticks([0.5, 1, 2, 4, 8, 12, 20, 30, 45, 100])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "12", "20", "30", "45", "100"],
                       fontsize=7.5)
    ax.axhline(-0.35, color="k", lw=0.8)
    ax.set_yticks(x); ax.set_yticklabels(df.ch, fontsize=8, style="italic")
    ax.set_xlabel("Passband of each dilation (Hz)")
    ax.set_title("G. Multi-scale frequency coverage")

    # ---------------- H  embedding geometry -----------------------------------
    ax = fig.add_subplot(gs[2, 3:6])
    ax.scatter(df["align"], df["unif"], s=55, color=NAVY, edgecolor="k", lw=0.6, zorder=3)
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
    ax.set_title("H. Embedding geometry vs targets")

    # ---------------- I  age information carried ------------------------------
    ax = fig.add_subplot(gs[3, 0:2])
    d = df.sort_values("r", ascending=False)
    ax.bar(range(len(d)), d.r.values, color=NAVY, edgecolor="k", lw=0.7)
    for i, row in enumerate(d.itertuples()):
        ax.text(i, row.r + 0.012, f"{row.slope:+.2f}", ha="center", fontsize=6.2,
                color=RED, rotation=90)
    style_xticks(ax, d.ch.values)
    ax.set_ylim(0, d.r.max() * 1.26)
    ax.set_xlabel("Channel"); ax.set_ylabel("Subject-level age correlation  r")
    ax.set_title("I. Age information carried (red = slope)")

    # ---------------- J-K  final training curves ------------------------------
    chans = [c for c in df.ch if c in hist]
    ramp = plt.cm.coolwarm(np.linspace(0, 1, max(len(chans), 2)))

    axJ = fig.add_subplot(gs[3, 2:4])
    axK = fig.add_subplot(gs[3, 4:6])
    if not chans:
        for ax, t in ((axJ, "J. Training loss by channel"),
                      (axK, "K. Validation score by epoch")):
            ax.text(0.5, 0.5, "training not finished", ha="center", va="center",
                    fontsize=10, color=GRAY, transform=ax.transAxes)
            ax.set_title(t)
    else:
        for i, ch in enumerate(chans):
            h = hist[ch]["history"]
            ep = [p["epoch"] for p in h]
            axJ.plot(ep, [p["train_loss"] for p in h], lw=1.2, color=ramp[i], label=ch)
            axK.plot(ep, [-p["score"] for p in h], lw=1.2, color=ramp[i], label=ch)
            be = hist[ch]["best_epoch"]
            if 0 < be <= len(h):
                axK.plot(be, -h[be - 1]["score"], marker="o", ms=5, color=ramp[i],
                         markeredgecolor="k", markeredgewidth=0.6, zorder=4)
        axJ.set_xlabel("Epoch"); axJ.set_ylabel("Contrastive training loss")
        axJ.set_title("J. Training loss by channel")
        axK.set_xlabel("Epoch"); axK.set_ylabel("Validation information score")
        axK.set_title("K. Validation score by epoch (dot = best)")
        axK.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.9)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig3_stage2_encoders.{ext}"),
                    dpi=190, bbox_inches="tight")
    print("saved", os.path.join(RES, "Fig3_stage2_encoders.pdf"))

    # --------------------------------------------------------- summary table
    tab = df.copy()
    tab["dilations"] = tab.dil.apply(lambda d: "/".join(map(str, d)))
    tab["passband_Hz"] = tab.apply(lambda r: f"{r.f_lo:.1f}-{r.f_hi:.0f}", axis=1)
    tab["params"] = tab.conv_par + tab.gru_par
    cols = ["ch", "lobe", "n_trials", "k", "band_set", "dilations", "n_dilations",
            "passband_Hz", "rf_ms", "stride", "conv_ch", "conv_layers",
            "branch_mix", "gru", "emb", "params", "r", "knn", "slope", "mae",
            "sd_ratio", "align", "unif", "score"]
    out = tab[cols].round(3)
    out.to_csv(os.path.join(RES, "Table_encoders_stage2.csv"), index=False)

    def to_md(d):
        head = "| " + " | ".join(d.columns) + " |"
        rule = "|" + "|".join("---" for _ in d.columns) + "|"
        body = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in d.itertuples(index=False)]
        return "\n".join([head, rule] + body)

    with open(os.path.join(RES, "Table_encoders_stage2.md"), "w") as f:
        f.write("# Selected per-channel encoders (stage-2 search)\n\n")
        f.write("`n_dilations` = how many distinct dilations the band set "
                "produces; each gets its own Conv1d over the same input.\n"
                "`branch_mix` is the literal Optuna parameter name recorded in the "
                "study database: `concat` keeps all dilation outputs, `proj` adds a "
                "1x1 conv to compress them before the GRU.\n\n")
        f.write(to_md(out))
        f.write("\n\nTrained on 718 public subjects; the 81 in-house subjects are "
                "held out as the transfer cohort.\n")
        f.write(f"\nNo-information subject MAE floor: 12.88 yr. "
                f"Mean r {df.r.mean():.3f}, mean slope {df.slope.mean():+.3f}, "
                f"mean subject MAE {df.mae.mean():.2f} yr.\n")
    print("saved", os.path.join(RES, "Table_encoders_stage2.csv"))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
