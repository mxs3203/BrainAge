"""What each channel's dilated conv branches are tuned to, for the t25 encoders.

USAGE: python figures_convfilters_t25.py
"""
import os, sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# the checkpoint is a pickled module, so its defining module must be importable
sys.path.insert(0, os.path.join(HERE, "strengthen"))
os.environ.setdefault("HARMONIZED_F", "1")
FS = 256.0
CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
LOBE = {"AF3":"frontal","F7":"frontal","F3":"frontal","FC5":"frontal","FC6":"frontal",
        "F4":"frontal","F8":"frontal","AF4":"frontal","T7":"temporal","T8":"temporal",
        "P7":"posterior","P8":"posterior","O1":"posterior","O2":"posterior"}
LC = {"frontal": "#1F4E79", "temporal": "#B8860B", "posterior": "#C62828"}


def branch_filters(conv, nfft=2048):
    """|H(f)| per individual filter. Averaging across filters flattens the
    response, so keep them separate."""
    w = conv.weight.detach().cpu().numpy()         # [out, in, k]
    k, d = conv.kernel_size[0], conv.dilation[0]
    L = (k - 1) * d + 1
    imp = np.zeros((w.shape[0], L))
    imp[:, ::d] = w[:, 0, :]                       # in_channels is 1 here
    H = np.abs(np.fft.rfft(imp, n=nfft, axis=1))   # [out, nfreq]
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return f, H


def main():
    m = torch.load(os.path.join(RES, "model_age_phase4_t25_ep10.pt"),
                   map_location="cpu", weights_only=False)
    fig = plt.figure(figsize=(17.0, 6.0))
    gs = fig.add_gridspec(2, 7, hspace=0.62, wspace=0.30)
    rows = []
    for i, c in enumerate(CH):
        ax = fig.add_subplot(gs[i // 7, i % 7])
        enc = m.encoders[i]
        peaks, allcent = [], []
        for bi, br in enumerate(enc.branches):
            conv = next(mm for mm in br.modules()
                        if isinstance(mm, torch.nn.Conv1d) and mm.in_channels == 1)
            f, H = branch_filters(conv)
            keep = (f >= 1) & (f <= 45)
            fk, Hk = f[keep], H[:, keep]
            cent = (fk[None, :] * Hk).sum(1) / (Hk.sum(1) + 1e-12)   # per filter
            pkf = fk[np.argmax(Hk, axis=1)]
            allcent.append(cent)
            x = bi + np.linspace(-0.26, 0.26, len(cent))
            ax.scatter(x, pkf, s=7, color=LC[LOBE[c]], alpha=0.55,
                       edgecolor="none", zorder=3)
            ax.plot([bi - 0.3, bi + 0.3], [np.median(pkf)] * 2, color="#333333",
                    lw=2.0, zorder=4)
            peaks.append((conv.dilation[0], conv.kernel_size[0],
                          float(np.median(pkf)), float(pkf.std())))
        ax.set_xticks(range(len(enc.branches)))
        ax.set_xticklabels([f"d{b}" for b, _, _, _ in peaks], fontsize=7.5)
        ax.set_ylim(0, 48)
        ax.set_title(c, fontsize=10.5, color=LC[LOBE[c]], fontweight="bold", pad=4)
        ax.tick_params(labelsize=7.5)
        if i % 7 == 0:
            ax.set_ylabel("filter peak freq (Hz)", fontsize=8.5)
        if i // 7 == 1:
            ax.set_xlabel("branch (dilation)", fontsize=8.5)
        ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)
        rows.append((c, LOBE[c], peaks))
    fig.text(0.5, 0.99, "Peak frequency of every learned filter, by channel and "
             "dilation branch (bar = median)", ha="center", fontsize=12)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        fig.savefig(os.path.join(RES, f"SuppFig_conv_filters_t25.{ext}"),
                    bbox_inches="tight", facecolor="white", **kw)
    print("wrote results/SuppFig_conv_filters_t25.pdf (+ .png)\n")
    print(f"{'ch':5s}{'lobe':11s} branch peaks (dilation, kernel -> peak Hz)")
    for c, lb, pk in rows:
        s = "  ".join(f"d{d}: {p:5.1f}+-{sd:4.1f}Hz" for d, k, p, sd in pk)
        print(f"{c:5s}{lb:11s}{s}")


if __name__ == "__main__":
    main()
