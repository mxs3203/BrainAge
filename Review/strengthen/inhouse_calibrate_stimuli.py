"""In-house calibration and a per-stimulus report."""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("HARMONIZED_F", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import channel_encoder_search_arch as A  # noqa: E402
import config_review as C  # noqa: E402
from downstream_models import (compute, subject_tables, subject_tables_masked,  # noqa: E402
                               feat_concat, fit_ridge, metrics)
from stimulus_groups import group as sgroup  # noqa: E402
from inhouse_within_cv import cv_predict, models  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
SPLIT = os.path.join(HERE, "final_encoders", "inhouse_split.json")
N_PERM = 5000


def fit_public(tabs, ages, chans):
    """Public model: concat embeddings -> ridge, 5 folds, averaged at predict."""
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(ages))
    X = feat_concat(tabs, chans)
    fitted = []
    oof = np.empty(len(ages))
    for te in np.array_split(order, 5):
        tr = np.setdiff1d(np.arange(len(ages)), te)
        oof[te], mdl = fit_ridge(X[tr], ages[tr], X[te])
        fitted.append(mdl)
    pub = metrics(ages, oof)
    return (lambda t: np.mean([m.predict(sc.transform(feat_concat(t, chans)))
                               for sc, m in fitted], 0)), pub


def main():
    P, I = compute()
    ptabs, pages, pids, _ = subject_tables(P)
    chans = [c for c in A.CHANNELS if c in ptabs]
    predict, pub = fit_public(ptabs, pages, chans)
    print(f"public model: r {pub['r']:.3f} MAE {pub['mae']:.2f} "
          f"slope {pub['slope']:+.3f}\n")

    sel = set(json.load(open(SPLIT))["select"])
    stim = np.asarray(I["stim"]).astype(str)
    ec = np.array([sgroup("ours", x) == "rest_ec" for x in stim])
    rows = []

    # ---------- 1. calibration, fitted on SELECT, applied to LOCKED ----------
    print("1. CALIBRATION  (gain fitted on select 40, applied to locked 37)")
    for cname, mask in (("eyes-closed (incl phase_15)", ec),
                        ("all stimuli pooled", np.ones(len(stim), bool))):
        t, a, ids_, _ = subject_tables_masked(I, mask)
        p = predict(t)
        issel = np.array([s in sel for s in ids_])
        if issel.sum() < 10 or (~issel).sum() < 10:
            continue
        k = np.polyfit(p[issel], a[issel], 1)          # fitted on SELECT only
        for tag, keep in (("select (in-sample)", issel), ("LOCKED (held-out)", ~issel)):
            raw = metrics(a[keep], p[keep])
            cal = metrics(a[keep], np.polyval(k, p[keep]))
            for how, m in (("raw", raw), ("calibrated", cal)):
                rows.append({"condition": cname, "subjects": tag, "n": int(keep.sum()),
                             "predictions": how, **{q: m[q] for q in
                                                    ("r", "rho", "mae", "slope")},
                             "pred_sd": float(np.std(
                                 p[keep] if how == "raw"
                                 else np.polyval(k, p[keep])))})
            print(f"   {cname:28s} {tag:18s} n{int(keep.sum()):3d} | "
                  f"raw slope {raw['slope']:+.3f} MAE {raw['mae']:6.2f} "
                  f"-> cal slope {cal['slope']:+.3f} MAE {cal['mae']:6.2f} "
                  f"| r {raw['r']:+.3f} (unchanged: {cal['r']:+.3f})", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.round(4).to_csv(os.path.join(RES, "Table_inhouse_calibration.csv"), index=False)

    # ---------- 2. data quantity: within-cohort CV, all stimuli -------------
    print("\n2. DATA QUANTITY  (within-cohort CV on select 40, all stimuli pooled)")
    t, a, ids_, _ = subject_tables_masked(I, np.ones(len(stim), bool))
    keep = np.array([s in sel for s in ids_])
    X, y = feat_concat(t, chans)[keep], a[keep]
    floor = float(np.abs(y - np.median(y)).mean())
    nseg_ec = 12
    print(f"   {len(y)} subjects, chance MAE {floor:.2f} yr "
          f"(eyes-closed only gave ~{nseg_ec} independent segments/subject)")
    rng = np.random.default_rng(C.SEED)
    qrows = []
    for name, mdl in models().items():
        pr = cv_predict(mdl, X, y)
        m = metrics(y, pr)
        null = np.array([np.corrcoef(rng.permutation(y), pr)[0, 1]
                         for _ in range(N_PERM)]) if np.std(pr) > 1e-9 else np.zeros(1)
        pv = (1 + (null >= m["r"]).sum()) / (1 + N_PERM)
        qrows.append({"model": name, **{q: m[q] for q in ("r", "rho", "mae", "slope")},
                      "p_perm": pv})
        print(f"   {name:11s} r {m['r']:+.3f} MAE {m['mae']:6.2f} "
              f"slope {m['slope']:+.3f} p {pv:.4f}", flush=True)
    qd = pd.DataFrame(qrows)
    print(f"   -> beat null: {(qd.p_perm < 0.05).sum()}/{len(qd)}  "
          f"(eyes-closed only was 0/5)")

    # ---------- 3. per-stimulus report --------------------------------------
    print("\n3. PER-STIMULUS")
    srows = []
    for st in sorted(set(stim)):
        m_ = stim == st
        if m_.sum() < 100:
            continue
        t, a, ids_, _ = subject_tables_masked(I, m_)
        if len(a) < 20 or np.std(a) < 2:
            continue
        p = predict(t)
        mm = metrics(a, p)
        null = np.array([np.corrcoef(rng.permutation(a), p)[0, 1]
                         for _ in range(1000)]) if np.std(p) > 1e-9 else np.zeros(1)
        srows.append({"stimulus": st, "n": len(a), **{q: mm[q] for q in
                                                      ("r", "rho", "mae", "slope")},
                      "p_perm": (1 + (null >= mm["r"]).sum()) / 1001,
                      "group": sgroup("ours", st)})
    sd = pd.DataFrame(srows).sort_values("r", ascending=False)
    # Benjamini-Hochberg across the stimulus family
    m_tot = len(sd)
    sd["bh_thresh"] = (np.arange(1, m_tot + 1) / m_tot) * 0.05
    sd_sorted = sd.sort_values("p_perm").reset_index(drop=True)
    sd_sorted["bh_thresh"] = (np.arange(1, m_tot + 1) / m_tot) * 0.05
    sd_sorted["bh_sig"] = sd_sorted.p_perm <= sd_sorted.bh_thresh
    nsig = int(sd_sorted.bh_sig.sum())
    sd_sorted.round(4).to_csv(
        os.path.join(RES, "Table_inhouse_per_stimulus_full.csv"), index=False)
    print(sd.head(12)[["stimulus", "n", "r", "rho", "mae", "slope",
                       "p_perm", "group"]].round(3).to_string(index=False))
    print(f"\n   {m_tot} stimuli tested | raw p<0.05: {(sd.p_perm < 0.05).sum()} "
          f"| surviving Benjamini-Hochberg at q=0.05: {nsig}")
    print(f"   best r {sd.r.max():+.3f} on {sd.iloc[0].stimulus} "
          f"(n={int(sd.iloc[0].n)}, p={sd.iloc[0].p_perm:.3f})")
    print(f"\nsaved 3 tables to {RES}")


if __name__ == "__main__":
    main()
