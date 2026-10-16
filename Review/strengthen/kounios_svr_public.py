"""Same classical models, but trained on the PUBLIC pool the encoders saw, then scored in-house.

This is the apples-to-apples version: our deep model pretrains on 718 public subjects,
so giving the classical pipeline only 40 in-house subjects was not a fair fight.
"""
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV, GroupKFold

RES = "../results"
def load(tag):
    z = np.load(os.path.join(RES, f"kounios_features{tag}.npz"), allow_pickle=True)
    return z["F"], z["y"], [str(s) for s in z["subs"]]

Fo, yo, so = load("")
F1, y1, s1 = load("_ds003775")
F2, y2, s2 = load("_ds005385")
pub_F = np.vstack([F1, F2]); pub_y = np.concatenate([y1, y2])
pub_id = [f"ds003775:{s}" for s in s1] + [f"ds005385:{s}" for s in s2]
pub_ds = np.array(["ds003775"] * len(s1) + ["ds005385"] * len(s2))

pr = pd.read_csv(os.path.join(RES, "Table_pred_joint_age_phase4_t25_oldmodel_newdata.csv"))
tr_ids = set(pr[pr.split == "public_train"].subject)
va_ids = set(pr[pr.split == "public_val"].subject)
locked = set(pr[pr.split == "ih_locked"].subject)

itr = [i for i, s in enumerate(pub_id) if s in tr_ids]
iva = [i for i, s in enumerate(pub_id) if s in va_ids]
ite = [i for i, s in enumerate(so) if s in locked]
print(f"public train {len(itr)} | public val {len(iva)} | in-house locked {len(ite)}")

Xtr, ytr = pub_F[itr], pub_y[itr]
Xva, yva = pub_F[iva], pub_y[iva]
Xte, yte = Fo[ite], yo[ite]
gtr = pub_ds[itr]
chance = np.abs(yte - np.median(ytr)).mean()
print(f"chance floor (public-train median on in-house ages): {chance:.3f} y\n")

cv = GroupKFold(n_splits=2)          # split by dataset: no dataset-identity shortcut
KS = [25, 100, 400, "all"]
GRIDS = {
 "SVR-RBF": (SVR(kernel="rbf"), {"m__C": [1, 10, 100, 1000],
    "m__gamma": ["scale", 1e-4, 1e-3], "m__epsilon": [1.0, 2.0, 5.0]}),
 "ElasticNet": (ElasticNet(max_iter=20000),
    {"m__alpha": [0.01, 0.1, 1.0, 10.0], "m__l1_ratio": [0.1, 0.5, 0.9]}),
 "GaussProc-RBF": (GaussianProcessRegressor(
     kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
     random_state=0), {}),
}
def mets(t, p):
    return (np.corrcoef(t, p)[0, 1], np.abs(p - t).mean(), np.polyfit(t, p - t, 1)[0])

rows, P = [], {}
for name, (est, grid) in GRIDS.items():
    pipe = Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_regression)), ("m", est)])
    g = dict(grid); g["sel__k"] = KS
    gs = GridSearchCV(pipe, g, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    pv = gs.predict(Xva); pt = gs.predict(Xte)
    P[name] = pt
    rv = mets(yva, pv); rt = mets(yte, pt)
    rows.append(dict(model=name, pub_val_r=round(rv[0], 4), pub_val_mae=round(rv[1], 3),
                     ih_r=round(rt[0], 4), ih_mae=round(rt[1], 3), ih_slope=round(rt[2], 3),
                     pct_vs_chance=round(100 * (1 - rt[1] / chance), 2)))
    print(f"{name:14s} public-val r {rv[0]:+.3f} MAE {rv[1]:5.2f} | "
          f"IN-HOUSE r {rt[0]:+.3f} MAE {rt[1]:5.2f} ({100*(1-rt[1]/chance):+6.2f}% vs chance)")
    print(f"               best {gs.best_params_}")

ens = np.mean([P[k] for k in P], 0)
rt = mets(yte, ens)
rows.append(dict(model="ensemble (mean of 3)", pub_val_r=np.nan, pub_val_mae=np.nan,
                 ih_r=round(rt[0], 4), ih_mae=round(rt[1], 3), ih_slope=round(rt[2], 3),
                 pct_vs_chance=round(100 * (1 - rt[1] / chance), 2)))
print(f"\n{'ensemble':14s} IN-HOUSE r {rt[0]:+.3f} MAE {rt[1]:5.2f} "
      f"({100*(1-rt[1]/chance):+6.2f}% vs chance)")

# paired bootstrap against our deep model on the identical 41 subjects
lk = pr[pr.split == "ih_locked"].set_index("subject")
ours = np.array([lk.loc[so[i], "p"] for i in ite])
assert np.allclose(np.array([lk.loc[so[i], "t"] for i in ite]), yte)
ae_o, ae_e = np.abs(ours - yte), np.abs(ens - yte)
rng = np.random.default_rng(0); n = len(yte); dm, dr = [], []
for _ in range(20000):
    b = rng.integers(0, n, n)
    dm.append(ae_e[b].mean() - ae_o[b].mean())
    dr.append(np.corrcoef(yte[b], ours[b])[0, 1] - np.corrcoef(yte[b], ens[b])[0, 1])
dm, dr = np.array(dm), np.array(dr)
print(f"\nours r {np.corrcoef(yte,ours)[0,1]:+.3f} MAE {ae_o.mean():.3f}")
print(f"public-trained classical ensemble vs ours:")
print(f"  dMAE {ae_e.mean()-ae_o.mean():+.3f} y  CI [{np.percentile(dm,2.5):+.3f},{np.percentile(dm,97.5):+.3f}] p={2*min((dm<=0).mean(),(dm>=0).mean()):.4f}")
print(f"  dr   {np.corrcoef(yte,ours)[0,1]-np.corrcoef(yte,ens)[0,1]:+.3f}  CI [{np.percentile(dr,2.5):+.3f},{np.percentile(dr,97.5):+.3f}] p={2*min((dr<=0).mean(),(dr>=0).mean()):.4f}")
pd.DataFrame(rows).to_csv(os.path.join(RES, "Table_kounios_public_trained.csv"), index=False)
np.savez(os.path.join(RES, "kounios_public_preds.npz"), yte=yte, ens=ens, ours=ours,
         **{k: v for k, v in P.items()})
print("\nwrote ../results/Table_kounios_public_trained.csv")
