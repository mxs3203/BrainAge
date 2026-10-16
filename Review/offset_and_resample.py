"""Does correcting the offset, or age-matching the training set, recover in-house"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
import channel_encoder_search_arch as A  # noqa: E402
import config_review as C  # noqa: E402
from downstream_models import (compute, subject_tables, subject_tables_masked,  # noqa: E402
                               feat_concat, metrics)
from stimulus_groups import group as sgroup  # noqa: E402

RES = os.path.join(HERE, "results")
BINS = np.array([17, 25, 32, 40, 48, 56, 75])


def ridge_fit(Xtr, ytr, w=None):
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    if w is None:
        m = RidgeCV(alphas=np.logspace(-2, 4, 14)).fit(sc.transform(Xtr), ytr)
    else:
        m = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr, sample_weight=w)
    return sc, m


def apply(model, X):
    sc, m = model
    return m.predict(sc.transform(X))


def cv_calibrate(y, p, kind, folds=5, seed=0):
    """Affine correction fitted out-of-fold within the in-house cohort."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    out = np.empty(len(y))
    for te in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, te)
        if kind == "intercept":
            b = np.mean(y[tr] - p[tr])
            out[te] = p[te] + b
        else:                                   # intercept + gain
            k = np.polyfit(p[tr], y[tr], 1)
            out[te] = np.polyval(k, p[te])
    return out


def match_weights(src_age, tgt_age):
    """Density-ratio weights so the source age histogram matches the target."""
    hs, _ = np.histogram(src_age, bins=BINS)
    ht, _ = np.histogram(tgt_age, bins=BINS)
    hs = np.maximum(hs, 1)
    ratio = (ht / max(ht.sum(), 1)) / (hs / hs.sum())
    b = np.clip(np.digitize(src_age, BINS[1:-1]), 0, len(ratio) - 1)
    w = ratio[b]
    return w / w.mean()


def match_subsample(src_age, tgt_age, rng):
    """Subsample source subjects so their age histogram matches the target."""
    ht, _ = np.histogram(tgt_age, bins=BINS)
    frac = ht / max(ht.sum(), 1)
    b = np.clip(np.digitize(src_age, BINS[1:-1]), 0, len(frac) - 1)
    # largest total n whose per-bin counts respect both teh target shape and supply
    cap = min((np.sum(b == i) / frac[i]) for i in range(len(frac)) if frac[i] > 0)
    keep = []
    for i in range(len(frac)):
        want = int(round(frac[i] * cap))
        avail = np.flatnonzero(b == i)
        if want and len(avail):
            keep.append(rng.choice(avail, min(want, len(avail)), replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.arange(len(src_age))


def main():
    P, I = compute()
    tabs, ages, ids, ds = subject_tables(P)
    chans = [c for c in A.CHANNELS if c in tabs]
    X = feat_concat(tabs, chans)
    ec = np.array([sgroup("ours", x) == "rest_ec" for x in I["stim"]])
    it, iages, _, _ = subject_tables_masked(I, ec)
    Xi = feat_concat(it, chans)
    rng = np.random.default_rng(C.SEED)

    print(f"public  n={len(ages)}  age {ages.mean():.1f} +- {ages.std():.1f}")
    print(f"in-house n={len(iages)}  age {iages.mean():.1f} +- {iages.std():.1f}")
    floor = float(np.abs(iages - np.median(iages)).mean())
    print(f"in-house no-information floor: {floor:.2f} yr\n")

    rows = []

    def record(name, p, note=""):
        m = metrics(iages, p)
        m["model"] = name
        m["note"] = note
        rows.append(m)
        print(f"  {name:34s} r {m['r']:+.3f} | MAE {m['mae']:6.2f} | "
              f"slope {m['slope']:+.3f}  {note}")

    # ---- baseline: trained on all public subjects -------------------------
    mdl = ridge_fit(X, ages)
    p0 = apply(mdl, Xi)
    print("baseline and calibration:")
    record("baseline (no correction)", p0)
    record("+ intercept only", cv_calibrate(iages, p0, "intercept"),
           "r unchanged by construction")
    record("+ intercept and gain", cv_calibrate(iages, p0, "affine"),
           "r unchanged by construction")

    # ---- resampling / reweighting the training set ------------------------
    print("\nage-matching the public training set to the in-house distribution:")
    keep = match_subsample(ages, iages, rng)
    mdl_s = ridge_fit(X[keep], ages[keep])
    record("subsampled to match age hist", apply(mdl_s, Xi),
           f"n={len(keep)} of {len(ages)}")
    w = match_weights(ages, iages)
    mdl_w = ridge_fit(X, ages, w=w)
    record("importance-weighted", apply(mdl_w, Xi), "all subjects kept")
    record("weighted + affine calibration",
           cv_calibrate(iages, apply(mdl_w, Xi), "affine"))

    df = pd.DataFrame(rows)[["model", "r", "mae", "slope", "note"]]
    os.makedirs(RES, exist_ok=True)
    df.round(4).to_csv(os.path.join(RES, "Table_offset_resample.csv"), index=False)
    print(f"\nin-house floor {floor:.2f} yr -- any MAE above it is worse than "
          f"predicting the median")
    best = df.loc[df.mae.idxmin()]
    print(f"lowest MAE: {best.model} at {best.mae:.2f} (r {best.r:+.3f})")
    print(f"highest r : {df.loc[df.r.idxmax()].model} at {df.r.max():+.3f}")


if __name__ == "__main__":
    main()
