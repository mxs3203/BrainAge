"""Supplementary Figure 4: EEG windows before and after contrastive augmentation.

Uses the real EEGAugmentorGPU and the frozen per-channel settings.

USAGE: python figures_augmentation.py [--channel P8]
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from utils.GpuAugmentor import EEGAugmentorGPU            # noqa: E402

OUT, RES = os.path.join(HERE, "Figures"), os.path.join(HERE, "results")
CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
FS = 256.0
BLUE, RED, GOLD, TEAL, GREY = "#1F4E79", "#C62828", "#B8860B", "#1B7F79", "#6E6E6E"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="P8")
    ap.add_argument("--window", type=int, default=1200)
    a = ap.parse_args()
    ci = CH.index(a.channel)

    h = pd.read_csv(os.path.join(RES, "SuppT1_frozen_hparams.csv")).set_index("channel")
    p = h.loc[a.channel]
    cfg = dict(noise=float(p.aug_noise), scale=float(p.aug_scale),
               shift=int(p.aug_shift), drop=int(p.aug_drop))
    print(f"{a.channel}: noise {cfg['noise']:.4f} x SD | scale +-{cfg['scale']:.3f} | "
          f"shift +-{cfg['shift']} samples | drop <= {cfg['drop']} samples")

    X = np.load(os.path.join(HERE, "strengthen", f"public_pool_ch{ci:02d}.npy"),
                mmap_mode="r")
    x0 = np.asarray(X[a.window], dtype=np.float32)
    t = np.arange(len(x0)) / FS

    torch.manual_seed(0)
    xb = torch.tensor(x0).unsqueeze(0)

    # each augmentation on its own, by disabling the others
    steps = []
    for name, kw in (("Gaussian noise", dict(noise_std_factor=cfg["noise"], scale_range=0.0,
                                             max_shift=0, max_drop_len=0)),
                     ("amplitude scaling", dict(noise_std_factor=0.0, scale_range=cfg["scale"],
                                                max_shift=0, max_drop_len=0)),
                     ("time shift", dict(noise_std_factor=0.0, scale_range=0.0,
                                         max_shift=cfg["shift"], max_drop_len=0)),
                     ("segment dropout", dict(noise_std_factor=0.0, scale_range=0.0,
                                              max_shift=0, max_drop_len=cfg["drop"]))):
        torch.manual_seed(3)
        steps.append((name, EEGAugmentorGPU(**kw)(xb)[0].numpy()))

    full = EEGAugmentorGPU(noise_std_factor=cfg["noise"], scale_range=cfg["scale"],
                           max_shift=cfg["shift"], max_drop_len=cfg["drop"])
    torch.manual_seed(1); v1 = full(xb)[0].numpy()
    torch.manual_seed(2); v2 = full(xb)[0].numpy()

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(3, 2, hspace=0.46, wspace=0.17,
                          height_ratios=[1.35, 1.0, 1.0])

    # A: each augmentation separately, stacked
    ax = fig.add_subplot(gs[0, :])
    off = 0.0
    step = 4.2 * np.std(x0)
    for k, (name, xa) in enumerate(steps):
        ax.plot(t, x0 - off, color=GREY, lw=0.8, alpha=0.55,
                label="original" if k == 0 else None)
        ax.plot(t, xa - off, color=[GOLD, TEAL, BLUE, RED][k], lw=1.0,
                label=name)
        ax.text(2.02, -off, name, va="center", fontsize=8.5,
                color=[GOLD, TEAL, BLUE, RED][k])
        off += step
    ax.set_xlim(0, 2.35); ax.set_yticks([])
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (a.u.)")
    ax.set_title(f"A. Each augmentation applied on its own ({a.channel}, grey = original)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # B/C: the two contrastive views
    for k, (v, lab) in enumerate(((v1, "view 1"), (v2, "view 2"))):
        ax = fig.add_subplot(gs[1, k])
        ax.plot(t, x0, color=GREY, lw=0.9, label="original")
        ax.plot(t, v, color=BLUE if k == 0 else RED, lw=0.9, label=lab)
        ax.set_xlim(0, 2); ax.set_xlabel("time (s)")
        if k == 0: ax.set_ylabel("amplitude (a.u.)")
        ax.set_title(f"{'BC'[k]}. Full augmentation, {lab}", fontsize=10.5, loc="left")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.16, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)

    # D: difference
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, v1 - x0, color=BLUE, lw=0.8, label="view 1 - original")
    ax.plot(t, v2 - x0, color=RED, lw=0.8, alpha=0.75, label="view 2 - original")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 2); ax.set_xlabel("time (s)"); ax.set_ylabel("difference (a.u.)")
    ax.set_title("D. Difference from the original", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.16, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    # E: spectra
    ax = fig.add_subplot(gs[2, 1])
    f = np.fft.rfftfreq(len(x0), 1 / FS); keep = (f >= 1) & (f <= 45)
    w = np.hanning(len(x0))
    for sig, c, lab in ((x0, GREY, "original"), (v1, BLUE, "view 1"), (v2, RED, "view 2")):
        P = np.abs(np.fft.rfft(sig * w))**2
        ax.semilogy(f[keep], P[keep], color=c, lw=1.1, label=lab)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (a.u.)")
    ax.set_title("E. Augmentation preserves the spectral profile", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.16, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        fig.savefig(os.path.join(OUT, f"SuppFig7.{ext}"), bbox_inches="tight",
                    facecolor="white", **kw)
    print("wrote Figures/SuppFig7.pdf (+ .png)")
    print(f"  corr(original, view1) = {np.corrcoef(x0, v1)[0,1]:.4f}")
    print(f"  corr(original, view2) = {np.corrcoef(x0, v2)[0,1]:.4f}")
    print(f"  corr(view1, view2)    = {np.corrcoef(v1, v2)[0,1]:.4f}")


if __name__ == "__main__":
    main()
