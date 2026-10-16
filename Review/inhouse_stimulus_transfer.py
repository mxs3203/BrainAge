"""Train on public eyes-closed rest, then test EACH in-house stimulus separately."""
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("HARMONIZED_F", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
from harmonized_io import build_windows, load_dataset   # noqa: E402
from stimulus_groups import group                       # noqa: E402

FS, WIN, STRIDE = 256.0, 512, 512
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 20), (20, 30), (30, 45)]
PER_SUBJ = 60
SOURCES = ["ds005385", "ds003775"]
MIN_SUBJ = 20


def rel_band(X):
    F = np.fft.rfft(X * np.hanning(X.shape[1])[None, :, None], axis=1)
    P = F.real ** 2 + F.imag ** 2
    f = np.fft.rfftfreq(X.shape[1], 1 / FS)
    B = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS], -1)
    B = B / np.maximum(B.sum(-1, keepdims=True), 1e-12)
    return np.log10(B + 1e-6).reshape(len(X), -1)


def subject_feats(ds, stimuli):
    w = build_windows(ds, recipe="prefiltered_uv", win=WIN, stride=STRIDE,
                      stimuli=stimuli, verbose=False)
    if w is None or len(w["X"]) < 100:
        return None
    sj = np.asarray(w["subject"]).astype(str)
    rng = np.random.default_rng(0)
    keep = []
    for u in np.unique(sj):
        i = np.flatnonzero(sj == u)
        keep.append(rng.choice(i, PER_SUBJ, replace=False) if len(i) > PER_SUBJ else i)
    k = np.sort(np.concatenate(keep))
    d = pd.DataFrame(rel_band(w["X"][k])); d["s"] = sj[k]
    M = d.groupby("s").mean()
    a = pd.Series(np.asarray(w["age"], float)[k]).groupby(pd.Series(sj[k])).first().loc[M.index]
    return M, a


def inhouse_split():
    _, recs = load_dataset("ours")
    subs = sorted(recs.subject.unique())
    age = np.array([float(recs[recs.subject == x].age.iloc[0]) for x in subs])
    o = np.argsort(age)
    return set(subs[i] for i in o[0::2]), set(subs[i] for i in o[1::2])


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    # ---- fit once on public eyes-closed rest -------------------------------
    Ms, As = [], []
    for ds in SOURCES:
        _, recs = load_dataset(ds)
        ec = [s for s in recs.stimulus.unique() if group(ds, s) == "rest_ec"]
        r = subject_feats(ds, ec)
        if r is None:
            continue
        Ms.append(r[0]); As.append(r[1])
        print(f"  source {ds:10s} {len(r[1]):4d} subjects, eyes-closed", flush=True)
    Mtr = pd.concat(Ms).to_numpy(); atr = pd.concat(As).to_numpy()
    sc = StandardScaler().fit(Mtr)
    mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(Mtr), atr)
    print(f"  probe fitted on {len(atr)} public subjects "
          f"(age {atr.min():.0f}-{atr.max():.0f})\n", flush=True)

    dev, lock = inhouse_split()
    _, recs = load_dataset("ours")
    rows = []
    for stim in sorted(recs.stimulus.unique()):
        r = subject_feats("ours", [stim])
        if r is None:
            continue
        M, a = r
        idx = np.array(M.index)
        for half, name in ((dev, "dev"), (lock, "lock")):
            m = np.array([i in half for i in idx])
            if m.sum() < MIN_SUBJ or np.std(a.to_numpy()[m]) < 2:
                continue
            p = mdl.predict(sc.transform(M.to_numpy()[m]))
            av = a.to_numpy()[m]
            rows.append({"stimulus": str(stim), "group": group("ours", stim),
                         "half": name, "n_subj": int(m.sum()),
                         "r": float(np.corrcoef(av, p)[0, 1]),
                         "mae": float(np.abs(p - av).mean()),
                         "slope": float(np.polyfit(av, p - av, 1)[0])})
    out = pd.DataFrame(rows)
    p_ = os.path.join(HERE, "results", "Table_inhouse_stimulus_transfer.csv")
    out.round(4).to_csv(p_, index=False)

    d = out[out.half == "dev"].sort_values("r", ascending=False)
    l = out[out.half == "lock"].set_index("stimulus")
    # permutation-free BH on a t-approximation of r, enough to show multiplicity
    from scipy import stats as st
    d = d.copy()
    d["p"] = [2 * (1 - st.t.cdf(abs(r) * np.sqrt((n - 2) / max(1e-9, 1 - r ** 2)), n - 2))
              for r, n in zip(d.r, d.n_subj)]
    pv = np.sort(d.p.values); n = len(pv); thr = 0.0
    for i, v in enumerate(pv, 1):
        if v <= 0.05 * i / n:
            thr = v
    d["fdr_sig"] = d.p <= thr

    print(f"{'stimulus':30s} {'group':8s} {'n':>4s} {'r_DEV':>7s} {'p':>7s} {'FDR':>4s} "
          f"{'r_LOCK':>7s} {'MAE_LOCK':>9s}")
    for t in d.head(15).itertuples():
        rl = l.loc[t.stimulus] if t.stimulus in l.index else None
        print(f"{t.stimulus[:30]:30s} {t.group:8s} {t.n_subj:4d} {t.r:+7.3f} {t.p:7.3f} "
              f"{'*' if t.fdr_sig else '':>4s} "
              f"{(f'{rl.r:+.3f}' if rl is not None else '  --'):>7s} "
              f"{(f'{rl.mae:.2f}' if rl is not None else '  --'):>9s}")
    print(f"\nBH-FDR 5% over {n} DEV stimuli -> threshold p<={thr:.4f}, "
          f"{int(d.fdr_sig.sum())} survive")
    both = d.merge(out[out.half == 'lock'][['stimulus', 'r']], on='stimulus',
                   suffixes=('_dev', '_lock'))
    if len(both) > 2:
        print(f"correlation between DEV and LOCK r across stimuli: "
              f"{np.corrcoef(both.r_dev, both.r_lock)[0,1]:+.3f}")
        print("  (near zero would mean the DEV ranking is noise and no choice is defensible)")


if __name__ == "__main__":
    main()
