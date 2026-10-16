"""Fit the Kounios-style models on OUR split: tune on the 40 held-in, score the 41 locked once."""
import os, sys
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

d = np.load(os.path.join(RES, "kounios_features.npz"), allow_pickle=True)
F, y, subs = d["F"], d["y"], list(d["subs"])
pred = pd.read_csv(os.path.join(RES, "Table_pred_joint_age_phase4_t25_oldmodel_newdata.csv"))
locked = set(pred[pred.split == "ih_locked"].subject)
heldin = set(pred[pred.split.isin(["ih_fit", "ih_val"])].subject)

itr = [i for i, s in enumerate(subs) if s in heldin]
ite = [i for i, s in enumerate(subs) if s in locked]
print(f"features {F.shape} | train {len(itr)} held-in | test {len(ite)} locked")
Xtr, ytr, Xte, yte = F[itr], y[itr], F[ite], y[ite]
chance = np.abs(yte - np.median(ytr)).mean()
print(f"chance floor (train median on test ages): {chance:.3f} y\n")

cv = KFold(5, shuffle=True, random_state=0)
GRIDS = {
 "SVR-RBF": (SVR(kernel="rbf"),
   {"m__C": [1, 10, 100, 1000], "m__gamma": ["scale", 1e-4, 1e-3],
    "m__epsilon": [1.0, 2.0, 5.0]}),
 "ElasticNet": (ElasticNet(max_iter=20000),
   {"m__alpha": [0.01, 0.1, 1.0, 10.0], "m__l1_ratio": [0.1, 0.5, 0.9]}),
 "GaussProc-RBF": (GaussianProcessRegressor(
     kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
     random_state=0), {}),
}
KS = [25, 100, 400, "all"]

def mets(t, p):
    return (np.corrcoef(t, p)[0, 1], np.abs(p - t).mean(),
            np.polyfit(t, p - t, 1)[0])

rows, preds = [], {}
for name, (est, grid) in GRIDS.items():
    pipe = Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_regression)), ("m", est)])
    g = dict(grid); g["sel__k"] = KS
    gs = GridSearchCV(pipe, g, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xtr, ytr)
    p = gs.predict(Xte)
    preds[name] = p
    r, mae, sl = mets(yte, p)
    cvmae = -gs.best_score_
    rows.append(dict(model=name, cv_mae=round(cvmae, 3), r=round(r, 4),
                     mae=round(mae, 3), slope=round(sl, 3),
                     pct_vs_chance=round(100 * (1 - mae / chance), 2),
                     beats_chance=mae < chance))
    print(f"{name:14s} cv_MAE {cvmae:5.2f} | test r {r:+.3f} MAE {mae:5.2f} "
          f"slope {sl:+.3f} | {100*(1-mae/chance):+6.2f}% vs chance")
    print(f"               best: {gs.best_params_}")

ens = np.mean([preds[k] for k in preds], 0)
r, mae, sl = mets(yte, ens)
rows.append(dict(model="ensemble (mean of 3)", cv_mae=np.nan, r=round(r, 4),
                 mae=round(mae, 3), slope=round(sl, 3),
                 pct_vs_chance=round(100 * (1 - mae / chance), 2),
                 beats_chance=mae < chance))
print(f"\n{'ensemble':14s} test r {r:+.3f} MAE {mae:5.2f} slope {sl:+.3f} | "
      f"{100*(1-mae/chance):+6.2f}% vs chance")

rows.append(dict(model="chance (train median)", cv_mae=np.nan, r=0.0,
                 mae=round(chance, 3), slope=-1.0, pct_vs_chance=0.0,
                 beats_chance=False))
rows.append(dict(model="OUR deep model (t25)", cv_mae=np.nan, r=0.5371,
                 mae=7.4397, slope=-0.701, pct_vs_chance=18.22,
                 beats_chance=True))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "Table_kounios_svr_baseline.csv"), index=False)
print("\n" + out.to_string(index=False))
print("\nwrote ../results/Table_kounios_svr_baseline.csv")
