"""Is there ANY age signal in the in-house cohort, read from the encoder embeddings?
Run on the SELECT 40 subjects only. The locked 37 are left untouched so they remain
"""
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
from downstream_models import (compute, subject_tables_masked, feat_concat,  # noqa: E402
                               metrics)
from stimulus_groups import group as sgroup  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
SPLIT = os.path.join(HERE, "final_encoders", "inhouse_split.json")
N_PERM = 5000
FOLDS = 8          # 40 subjects -> 5 per held-out fold


def models():
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    return {
        "ridge": RidgeCV(alphas=np.logspace(-3, 5, 25)),
        "elasticnet": ElasticNetCV(l1_ratio=[.3, .7, .95], n_alphas=40,
                                   max_iter=8000, random_state=0),
        "svr_rbf": SVR(C=10.0, gamma="scale", epsilon=2.0),
        "rf": RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                    random_state=0, n_jobs=8),
        "xgb": XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                            subsample=0.8, colsample_bytree=0.5,
                            min_child_weight=3, reg_lambda=5.0,
                            random_state=0, n_jobs=8, tree_method="hist"),
    }


def cv_predict(mdl, X, y, folds=FOLDS, seed=0):
    """Out-of-fold predictions. PCA is fitted inside each fold: with 40 subjets
    and thousands of features, fitting it on everything would leak."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    p = np.empty(len(y))
    for te in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, te)
        k = min(20, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=k).fit(X[tr])
        sc = StandardScaler().fit(pca.transform(X[tr]))
        m = clone(mdl).fit(sc.transform(pca.transform(X[tr])), y[tr])
        p[te] = m.predict(sc.transform(pca.transform(X[te])))
    return p


def main():
    _, I = compute()
    ec = np.array([sgroup("ours", x) == "rest_ec" for x in I["stim"]])
    tabs, ages, ids, _ = subject_tables_masked(I, ec)
    sel = set(json.load(open(SPLIT))["select"])
    keep = np.array([s in sel for s in ids])
    chans = [c for c in A.CHANNELS if c in tabs]
    X = feat_concat(tabs, chans)[keep]
    y = ages[keep]
    floor = float(np.abs(y - np.median(y)).mean())
    print(f"in-house SELECT: {len(y)} subjects, {X.shape[1]} embedding features")
    print(f"age mean {y.mean():.1f} SD {y.std():.2f} range {y.min():.0f}-{y.max():.0f}")
    print(f"chance MAE (median predictor) {floor:.2f} yr\n")
    print(f"{FOLDS}-fold CV, {N_PERM} label permutations per model:\n")

    rng = np.random.default_rng(C.SEED)
    rows = []
    for name, mdl in models().items():
        p = cv_predict(mdl, X, y)
        m = metrics(y, p)
        if np.std(p) > 1e-9:
            null = np.array([np.corrcoef(rng.permutation(y), p)[0, 1]
                             for _ in range(N_PERM)])
            pv = (1 + (null >= m["r"]).sum()) / (1 + N_PERM)
            nsd = float(null.std())
        else:
            pv, nsd = 1.0, float("nan")
        rows.append({"model": name, "r": m["r"], "rho": m["rho"], "mae": m["mae"],
                     "slope": m["slope"], "pred_sd": float(np.std(p)),
                     "null_sd": nsd, "p_perm": pv,
                     "beats_floor": bool(m["mae"] < floor)})
        print(f"  {name:11s} r {m['r']:+.3f} rho {m['rho']:+.3f} MAE {m['mae']:6.2f} "
              f"(floor {floor:.2f}) slope {m['slope']:+.3f} predSD {np.std(p):5.2f} "
              f"| null SD {nsd:.3f} p {pv:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.round(4).to_csv(os.path.join(RES, "Table_inhouse_within_cv.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))
    nsig = int((df.p_perm < 0.05).sum())
    print(f"\nmodels beating the permutation null at p<0.05: {nsig}/{len(df)}")
    print(f"models beating the no-information MAE floor:    {int(df.beats_floor.sum())}/{len(df)}")
    print(f"best r {df.r.max():+.3f} (p {df.loc[df.r.idxmax(),'p_perm']:.4f}), "
          f"best MAE {df.mae.min():.2f} vs floor {floor:.2f}")
    if nsig == 0:
        print("\n-> No model reads age from the in-house embeddings even when fitted"
              "\n   ON the in-house cohort. That is not a transfer failure, so no"
              "\n   change to the public-side search can fix it.")
    else:
        print("\n-> Age IS readable within the in-house cohort, so the public->in-house"
              "\n   gap is a genuine TRANSFER problem and worth attacking directly.")


if __name__ == "__main__":
    main()
