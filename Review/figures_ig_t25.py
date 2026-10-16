"""Attribution figures for t25: a 3-panel main figure and a heatmap supplement.

USAGE: python figures_ig_t25.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FS = 256.0
LOBE = {"AF3":"frontal","F7":"frontal","F3":"frontal","FC5":"frontal","FC6":"frontal",
        "F4":"frontal","F8":"frontal","AF4":"frontal","T7":"temporal","T8":"temporal",
        "P7":"posterior","P8":"posterior","O1":"posterior","O2":"posterior"}
LC = {"frontal": "#1F4E79", "temporal": "#B8860B", "posterior": "#C62828"}
BANDS = [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]


def boot_ci(v, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = np.array([v[r.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return np.percentile(m, [2.5, 97.5])


def main():
    z = np.load(os.path.join(RES, "ig_t25.npz"), allow_pickle=True)
    M, ch = z["M"], [str(c) for c in z["ch"]]
    bins = [str(b) for b in z["bins"]]
    W, Wage = z["W"], z["Wage"]              # [nwin, 14] mean|IG|, and ages
    nb, T, nc = M.shape

    # ---------------- MAIN: 3 panels ----------------
    fig = plt.figure(figsize=(16.5, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # A: attribution magnitude per channel, bootstrap CI over windows
    ax = fig.add_subplot(gs[0, 0])
    mag = W.mean(0)
    o = np.argsort(mag)[::-1]
    lo = np.array([boot_ci(W[:, i])[0] for i in o])
    hi = np.array([boot_ci(W[:, i])[1] for i in o])
    cols = [LC[LOBE[ch[i]]] for i in o]
    ax.bar(np.arange(nc), mag[o], 0.68, color=cols, edgecolor="white", linewidth=1.3,
           yerr=[mag[o] - lo, hi - mag[o]],
           error_kw=dict(ecolor="#444444", lw=1.1, capsize=3), zorder=3)
    ax.set_xticks(np.arange(nc))
    ax.set_xticklabels([ch[i] for i in o], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("mean |IG| attribution")
    ax.set_title("Attribution concentrates posteriorly", fontsize=10.5, pad=7)
    for lb, cc in LC.items():
        ax.scatter([], [], s=44, color=cc, label=lb)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
    ax.text(-0.13, 1.08, "A", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    # B: frequency content of the attributions, by lobe
    ax = fig.add_subplot(gs[0, 1])
    f = np.fft.rfftfreq(T, 1 / FS)
    keep = (f >= 1) & (f <= 45)
    for lb in ["frontal", "temporal", "posterior"]:
        idx = [i for i, c in enumerate(ch) if LOBE[c] == lb]
        P = np.abs(np.fft.rfft(M[:, :, idx], axis=1)).mean((0, 2))
        ax.plot(f[keep], P[keep], color=LC[lb], lw=2.0, label=lb)
    for _, blo, bhi in BANDS:
        ax.axvline(bhi, color="#CCCCCC", lw=0.7, zorder=0)
    y0, y1 = ax.get_ylim()
    for bi, (nm, blo, bhi) in enumerate(BANDS):
        # delta and theta are narrow; stagger so the labels do not collide
        yy = y1 * (0.975 if bi % 2 == 0 else 0.90)
        ax.text((blo + bhi) / 2, yy, nm, fontsize=7.5, ha="center", va="top",
                color="#777777")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("attribution spectrum")
    ax.set_title("Attribution is broadband, not band-limited", fontsize=10.5, pad=7)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    ax.text(-0.13, 1.08, "B", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    # C: attribution magnitude by age group, with CIs
    ax = fig.add_subplot(gs[0, 2])
    edges = [(20, 29), (30, 39), (40, 49), (50, 62)]
    for lb in ["frontal", "temporal", "posterior"]:
        idx = [i for i, c in enumerate(ch) if LOBE[c] == lb]
        mu, el, eh = [], [], []
        for a0, a1 in edges:
            k = (Wage >= a0) & (Wage <= a1)
            v = W[k][:, idx].mean(1)
            mu.append(v.mean()); c0, c1 = boot_ci(v); el.append(c0); eh.append(c1)
        mu = np.array(mu)
        ax.errorbar(np.arange(len(edges)), mu,
                    yerr=[mu - np.array(el), np.array(eh) - mu],
                    fmt="o-", color=LC[lb], lw=2.0, ms=8, mec="white", mew=1.3,
                    capsize=3, label=lb)
    ax.set_xticks(np.arange(len(edges))); ax.set_xticklabels(bins, fontsize=9)
    ax.set_xlabel("age group"); ax.set_ylabel("mean |IG| attribution")
    ax.set_title("Posterior dominance holds across age", fontsize=10.5, pad=7)
    ax.legend(fontsize=8.5, frameon=False)
    ax.text(-0.13, 1.08, "C", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        fig.savefig(os.path.join(RES, f"Fig_attribution_t25.{ext}"),
                    bbox_inches="tight", facecolor="white", **kw)
    print("wrote results/Fig_attribution_t25.pdf (+ .png)")

    # ---------------- SUPPLEMENT: the heatmaps ----------------
    NB = 64
    Mb = M.reshape(nb, NB, T // NB, nc).mean(2)
    vmax = np.percentile(np.abs(Mb), 99)
    f2 = plt.figure(figsize=(17.0, 5.6))
    g2 = f2.add_gridspec(2, 7, hspace=0.60, wspace=0.28)
    for i, c in enumerate(ch):
        ax = f2.add_subplot(g2[i // 7, i % 7])
        im = ax.imshow(Mb[:, :, i], aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, T / FS, -0.5, nb - 0.5],
                       interpolation="nearest")
        ax.set_yticks(range(nb)); ax.set_yticklabels(bins, fontsize=7.5)
        ax.set_xticks([0, 1, 2]); ax.tick_params(labelsize=7.5)
        ax.set_title(c, fontsize=10.5, color=LC[LOBE[c]], fontweight="bold", pad=4)
        if i % 7 == 0:
            ax.set_ylabel("age group", fontsize=8.5)
        if i // 7 == 1:
            ax.set_xlabel("time (s)", fontsize=8.5)
    cax = f2.add_axes([0.918, 0.22, 0.010, 0.56])
    cb = f2.colorbar(im, cax=cax)
    cb.set_label("mean IG attribution (red: pushes prediction older)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        f2.savefig(os.path.join(RES, f"SuppFig_attribution_heatmaps_t25.{ext}"),
                   bbox_inches="tight", facecolor="white", **kw)
    print("wrote results/SuppFig_attribution_heatmaps_t25.pdf (+ .png)")

    # ---------------- numbers for the text ----------------
    from scipy import stats as st
    post = [i for i, c in enumerate(ch) if LOBE[c] == "posterior"]
    front = [i for i, c in enumerate(ch) if LOBE[c] == "frontal"]
    u, pu = st.mannwhitneyu(mag[post], mag[front], alternative="greater")
    print(f"\nposterior mean |IG| {mag[post].mean():.4f}  frontal {mag[front].mean():.4f} "
          f"ratio {mag[post].mean()/mag[front].mean():.2f}  Mann-Whitney p={pu:.4f}")
    try:
        import pandas as pd
        t = pd.read_csv(os.path.join(RES, "Table_information_vs_chance.csv"))
        enc = t.set_index("ch")["r"].to_dict()
        v = np.array([enc[c] for c in ch])
        r_ = np.corrcoef(mag, v)[0, 1]
        rho = st.spearmanr(mag, v).statistic
        print(f"corr(mean|IG|, per-channel encoder r) = {r_:+.3f}  Spearman {rho:+.3f}")
    except Exception as e:
        print("encoder-r convergence unavailable:", e)
    for lb in ["frontal", "temporal", "posterior"]:
        idx = [i for i, c in enumerate(ch) if LOBE[c] == lb]
        v = W[:, idx].mean(1)
        r_ = np.corrcoef(Wage, v)[0, 1]
        print(f"  {lb:10s} mean|IG| vs age across windows: r={r_:+.3f}")
    np.savetxt(os.path.join(RES, "Table_attribution_channels.csv"),
               np.column_stack([np.arange(nc), mag]), delimiter=",",
               header="idx,mean_abs_ig", comments="")
    import pandas as pd
    pd.DataFrame({"channel": ch, "lobe": [LOBE[c] for c in ch],
                  "mean_abs_ig": mag}).sort_values("mean_abs_ig", ascending=False)\
        .to_csv(os.path.join(RES, "Table_attribution_channels.csv"), index=False)
    print("wrote results/Table_attribution_channels.csv")


if __name__ == "__main__":
    main()
