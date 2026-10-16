"""Classical feature + SVR baseline, in the style of Kounios et al. 2024 (Front Neuroergon).

The paper says 1,215 features over "power, connectivity, noise and variability"
for 9 bands at the 14 EPOC X electrodes but never gives the breakdown (1215 is
not a multiple of 126), so this is a reconstruction of the four families, not a
replication of their exact list. Counts are printed.

USAGE: python kounios_svr_baseline.py [--epoch 4.0] [--cache]
"""
import argparse, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
RES = os.path.join(REV, "results")
sys.path.insert(0, REV)
import harmonized_io as H

CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
# nine bands inside the 1-50 Hz range their recordings were filtered to
BANDS = [("delta",1,4),("theta",4,8),("alpha1",8,10),("alpha2",10,13),
         ("beta1",13,18),("beta2",18,25),("beta3",25,30),
         ("gamma1",30,40),("gamma2",40,50)]
PAIRS = [(i,j) for i in range(14) for j in range(i+1,14)]


def epochs_of(X, recs, nper, hop, cap=None):
    """Yield [nper, 14] epochs, never crossing a recording boundary."""
    k = 0
    for r in recs.itertuples():
        a, n = int(r.start), int(r.length)
        for s in range(a, a + n - nper + 1, hop):
            yield np.asarray(X[s:s + nper], dtype=np.float64)
            k += 1
            if cap and k >= cap:
                return


def subject_features(X, recs, fs, nper, cap=None):
    hop = nper // 2
    win = np.hanning(nper)
    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    idx = [np.where((freqs >= lo) & (freqs < hi))[0] for _, lo, hi in BANDS]
    fitband = np.where((freqs >= 1) & (freqs <= 50))[0]

    nep = 0
    Sxx = np.zeros((14, len(freqs)))
    Sxy = np.zeros((len(PAIRS), len(freqs)), dtype=np.complex128)
    bp_ep = []                                  # per-epoch band power, for variability
    for ep in epochs_of(X, recs, nper, hop, cap):
        ep = ep - ep.mean(0)
        F = np.fft.rfft(ep * win[:, None], axis=0).T     # [14, nfreq]
        p = (F.real**2 + F.imag**2)
        Sxx += p
        for k, (i, j) in enumerate(PAIRS):
            Sxy[k] += F[i] * np.conj(F[j])
        bp_ep.append([p[:, ix].mean(1) for ix in idx])    # [9][14]
        nep += 1
    if nep < 4:
        return None, nep
    Sxx /= nep; Sxy /= nep
    bp_ep = np.asarray(bp_ep)                   # [nep, 9, 14]

    feats, names = [], []

    # 1. absolute band power, log -- 14 x 9
    bp = np.stack([Sxx[:, ix].mean(1) for ix in idx])     # [9, 14]
    feats.append(np.log10(bp + 1e-20).ravel())
    names += [f"pow_{b}_{c}" for b, _, _ in BANDS for c in CH]

    # 2. relative band power -- 14 x 9
    tot = bp.sum(0, keepdims=True)
    feats.append((bp / (tot + 1e-20)).ravel())
    names += [f"relpow_{b}_{c}" for b, _, _ in BANDS for c in CH]

    # 3. connectivity: magnitude-squared coherence -- 91 x 9
    coh = np.empty((len(BANDS), len(PAIRS)))
    for k, (i, j) in enumerate(PAIRS):
        num = np.abs(Sxy[k])**2
        den = Sxx[i] * Sxx[j] + 1e-20
        c = num / den
        for bi, ix in enumerate(idx):
            coh[bi, k] = c[ix].mean()
    feats.append(coh.ravel())
    names += [f"coh_{b}_{CH[i]}-{CH[j]}" for b, _, _ in BANDS for (i, j) in PAIRS]

    # 4. variability: coefficient of variation of band power over epochs -- 14 x 9
    cv = bp_ep.std(0) / (bp_ep.mean(0) + 1e-20)
    feats.append(cv.ravel())
    names += [f"cv_{b}_{c}" for b, _, _ in BANDS for c in CH]

    # 5. noise: residual of the log-log PSD from its 1/f linear fit, per band.
    # captures how far each band sits off the aperiodic backbone.
    lf = np.log10(freqs[fitband] + 1e-12)
    resid = np.empty((len(BANDS), 14))
    for c in range(14):
        lp = np.log10(Sxx[c, fitband] + 1e-20)
        sl, ic = np.polyfit(lf, lp, 1)
        rr = lp - (sl * lf + ic)
        full = np.full(len(freqs), np.nan); full[fitband] = rr
        for bi, ix in enumerate(idx):
            resid[bi, c] = np.nanmean(full[ix])
    feats.append(resid.ravel())
    names += [f"noise_{b}_{c}" for b, _, _ in BANDS for c in CH]

    return np.concatenate(feats), names, nep


def build(args):
    cache = os.path.join(HERE, f"kounios_features_{args.dataset}.pkl")
    if args.cache and os.path.exists(cache):
        print("using cached features")
        return pickle.load(open(cache, "rb"))

    X, recs = H.load_dataset(args.dataset)
    fs = H.info(args.dataset)["fs"]
    nper = int(round(args.epoch * fs))
    print(f"{args.dataset}: {X.shape} @ {fs} Hz | epoch {nper} samples ({args.epoch}s)")

    rows, ages, subs, names = [], [], [], None
    for sub, g in recs.groupby("subject", sort=True):
        out = subject_features(X, g, fs, nper, args.max_epochs)
        if out[0] is None:
            print(f"  skip {sub}: only {out[1]} epochs"); continue
        f, nm, nep = out
        names = names or nm
        rows.append(f); ages.append(float(g.age.iloc[0])); subs.append(sub)
        print(f"  {sub:8s} {nep:5d} epochs  {len(f):5d} features", flush=True)
    D = (np.asarray(rows), np.asarray(ages), subs, names)
    pickle.dump(D, open(cache, "wb"))
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=float, default=4.0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--dataset", default="ours")
    ap.add_argument("--max-epochs", type=int, default=400)
    a = ap.parse_args()
    F, y, subs, names = build(a)
    print(f"\nfeature matrix {F.shape}  ({len(subs)} subjects)")
    np.savez(os.path.join(RES, f"kounios_features_{a.dataset}.npz"),
             F=F, y=y, subs=np.array(subs), names=np.array(names))
    print(f"wrote ../results/kounios_features_{a.dataset}.npz")
