"""Does the preprocessing preserve the spectrum the later analyses use?

Reviewer 1 point 11. Four preprocessing recipes are applied to the same
eyes-closed recordings and compared on: the mean spectrum, spectral homogeneity
across frequencies (how smoothly the 14-channel topography changes from one
frequency bin to the next), split-half reliability of the topography across
participants, and the downstream anterior/posterior age index.
USAGE: python figures_spectral_quality.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# deliberately NOT HARMONIZED_F: start from the unfiltered harmonised array so
# each recipe does its own filtering
os.environ.pop("HARMONIZED_F", None)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "Figures")

from harmonized_io import build_windows, load_dataset  # noqa: E402
from stimulus_groups import group as sgroup  # noqa: E402
from pasa_age_groups import CH, log_ratio, perm_p, paper_bands  # noqa: E402
from pasa_index import _pearson  # noqa: E402
from pasa_age_groups import _letter  # noqa: E402

FS = 256.0
ANT = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6"]
POS = ["P7", "P8", "O1", "O2"]
ix = lambda n: np.array([CH.index(c) for c in n])
# (label, recipe, colour, linestyle). raw and filt_only draw on top of each
# other -- teh harmonised array is already common-average referenced and the
# 0.5-45 Hz filter is flat in-band -- so they need distinguishable dashes.
RECIPES = [("raw (no filter, no artefact rejection)", "raw", "#7f7f7f", "-"),
           ("filter only, no artefact rejection", "filt_only", "#1f3864", "--"),
           ("published pipeline", "filtered", "#c00000", "-"),
           ("strict (1 Hz HP, 75 uV)", "strict", "#2E7D5B", ":")]
CACHE = os.path.join(RES, "spectral_quality.npz")


def psd_per_subject(X, subj, nmax=200):
    """[S,14,F] mean log10 power per participant, and the frequency axis."""
    nt = X.shape[1]
    f = np.fft.rfftfreq(nt, 1.0 / FS)
    w = np.hanning(nt)
    keep = (f >= 1) & (f <= 55)
    subs = sorted(set(subj))
    out = np.zeros((len(subs), 14, keep.sum()))
    for i, s in enumerate(subs):
        idx = np.flatnonzero(subj == s)
        if len(idx) > nmax:
            idx = idx[np.linspace(0, len(idx) - 1, nmax).astype(int)]
        xb = np.asarray(X[idx], dtype=np.float64).transpose(0, 2, 1)   # [n,14,T]
        xb = xb - xb.mean(-1, keepdims=True)
        P = (np.abs(np.fft.rfft(xb * w, axis=-1)) ** 2).mean(0)        # [14,F]
        out[i] = np.log10(P[:, keep] + 1e-20)
    return np.array(subs), out, f[keep]


def topo(P):
    """Per-frequency topography: z-scored across the 14 channels."""
    m = P.mean(1, keepdims=True)
    s = P.std(1, keepdims=True) + 1e-12
    return (P - m) / s


def homogeneity(P):
    """Mean across-participant correlation between topographies at f and f'."""
    T = topo(P)                                   # [S,14,F]
    S, _, F = T.shape
    H = np.zeros((F, F))
    for i in range(S):
        H += np.corrcoef(T[i].T)
    return H / S


def split_half(P, rng):
    """Correlation of the mean topography between two halves of the cohort."""
    S, _, F = P.shape
    idx = rng.permutation(S)
    a, b = idx[: S // 2], idx[S // 2:]
    Ta, Tb = topo(P[a]).mean(0), topo(P[b]).mean(0)
    return np.array([np.corrcoef(Ta[:, k], Tb[:, k])[0, 1] for k in range(F)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        data = z["data"].item()
        freqs, age = z["freqs"], z["age"]
    else:
        _, recs = load_dataset("ours")
        ec = sorted({s for s in recs.stimulus.unique()
                     if sgroup("ours", s) == "rest_ec"})
        data, age, freqs = {}, None, None
        for lab, rec, _ in RECIPES:
            w = build_windows("ours", recipe=rec, win=512, stride=a.stride,
                              stimuli=ec, verbose=False)
            subj = np.asarray(w["subject"]).astype(str)
            subs, P, freqs = psd_per_subject(w["X"].astype(np.float32), subj)
            if age is None:
                am = {s: float(np.asarray(w["age"])[subj == s][0]) for s in subs}
                age = np.array([am[s] for s in subs])
            data[rec] = P
            print(f"{lab:26s} {w['X'].shape[0]:6d} windows  PSD {P.shape}",
                  flush=True)
        np.savez_compressed(CACHE, data=np.array(data, dtype=object),
                            freqs=freqs, age=age)

    BANDS, BS, _ = paper_bands()
    an, po = ix(ANT), ix(POS)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(12.6, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    # ---- A  mean spectrum -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for lab, rec, c, ls in RECIPES:
        ax.plot(freqs, data[rec].mean((0, 1)), color=c, lw=1.8, ls=ls,
                label=lab)
    ax.axvspan(49, 51, color="k", alpha=.08)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("log10 power (mean)")
    _letter(ax, "A")
    ax.legend(fontsize=7, frameon=False)

    # ---- B  adjacent-frequency topography correlation ---------------
    ax = fig.add_subplot(gs[0, 1])
    rows = []
    for lab, rec, c, ls in RECIPES:
        H = homogeneity(data[rec])
        adj = np.diag(H, 1)
        ax.plot(freqs[:-1], adj, color=c, lw=1.7, ls=ls, label=lab)
        rows.append(dict(recipe=lab, homogeneity_mean=round(float(adj.mean()), 3),
                         homogeneity_min=round(float(adj.min()), 3)))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("topography r, f vs f+0.5 Hz")
    _letter(ax, "B")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    # ---- D  full homogeneity matrix, published pipeline -------------
    ax = fig.add_subplot(gs[1, 1])
    H = homogeneity(data["filtered"])
    im = ax.imshow(H, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]])
    fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("frequency (Hz)")
    _letter(ax, "D")

    # ---- D  split-half reliability ----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for lab, rec, c, ls in RECIPES:
        sh = np.mean([split_half(data[rec], np.random.default_rng(i))
                      for i in range(50)], axis=0)
        ax.plot(freqs, sh, color=c, lw=1.7, ls=ls, label=lab)
        for r in rows:
            if r["recipe"] == lab:
                r["split_half_mean"] = round(float(sh.mean()), 3)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("split-half topography r")
    _letter(ax, "C")
    ax.set_ylim(-.1, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    # panels E (band-power agreement) and F (index per recipe) were cut; their
    # numbers are still written to the table so the text can quote them
    ref = data["filtered"]
    an_, po_ = an, po
    for lab, rec, c, ls in RECIPES:
        bs = []
        for b in BS:
            lo, hi = BANDS[b]
            m = (freqs >= lo) & (freqs < hi)
            if m.any():
                bs.append(data[rec][:, :, m].mean(-1))
        # paper reports the log difference, not exp(d) -- same scale here or
        # the recipe comparison isn't comparable to Figure 6
        _, R = log_ratio(np.mean(bs, axis=0), an_, po_)
        r = _pearson(R, age)
        pp_ = perm_p(R, age, np.random.default_rng(11), a.nperm)
        bandr = []
        for b, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            if m.any():
                bandr.append(_pearson(data[rec][:, :, m].mean(-1).ravel(),
                                      ref[:, :, m].mean(-1).ravel()))
        for q in rows:
            if q["recipe"] == lab:
                q.update(antpos_r=round(float(r), 3),
                         antpos_p=round(float(pp_), 4),
                         bandpower_r_vs_published=round(float(np.mean(bandr)), 3))

    fig.suptitle("Spectral quality control: preprocessing preserves the "
                 "spectrum used downstream\n(81 participants, eyes-closed rest)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"SuppFig5.{ext}"),
                    dpi=165, bbox_inches="tight")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "Table_spectral_quality.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print("\nwrote Figures/SuppFig5.pdf (+ .png)")


if __name__ == "__main__":
    main()
