"""Supplement figure for the classical-pipeline comparison. 6 panels, A-F.

USAGE: python figures_classical_supp.py
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BLUE, RED, GOLD, TEAL, GREY = "#1F4E79", "#C62828", "#B8860B", "#1B7F79", "#6E6E6E"
CACHE = os.path.join(RES, "classical_supp_cache.npz")

GRIDS = {
 "SVR": (SVR(kernel="rbf"), {"m__C": [1, 10, 100, 1000],
    "m__gamma": ["scale", 1e-4, 1e-3], "m__epsilon": [1.0, 2.0, 5.0]}),
 "ElasticNet": (ElasticNet(max_iter=20000),
    {"m__alpha": [0.01, 0.1, 1.0, 10.0], "m__l1_ratio": [0.1, 0.5, 0.9]}),
 "GaussProc": (GaussianProcessRegressor(
    kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
    random_state=0), {}),
}
FAMS = [("pow_", "abs power"), ("relpow_", "rel power"), ("coh_", "coherence"),
        ("cv_", "variability"), ("noise_", "1/f residual")]


def mets(t, p):
    return (np.corrcoef(t, p)[0, 1], np.abs(p - t).mean(), np.polyfit(t, p - t, 1)[0])


def compute():
    if os.path.exists(CACHE):
        print("using cached panel data")
        return dict(np.load(CACHE, allow_pickle=True))
    z = np.load(os.path.join(RES, "kounios_features.npz"), allow_pickle=True)
    F, y, subs, names = z["F"], z["y"], [str(s) for s in z["subs"]], [str(n) for n in z["names"]]
    pr = pd.read_csv(os.path.join(RES, "Table_pred_joint_age_phase4_t25_oldmodel_newdata.csv"))
    lk = pr[pr.split == "ih_locked"].set_index("subject")
    locked = set(lk.index); heldin = set(pr[pr.split.isin(["ih_fit", "ih_val"])].subject)
    ite = [i for i, s in enumerate(subs) if s in locked]
    ihi = [i for i, s in enumerate(subs) if s in heldin]
    Xtr, ytr, Xte, yte = F[ihi], y[ihi], F[ite], y[ite]
    cv = KFold(5, shuffle=True, random_state=0)

    pt = {}
    for k, (est, grid) in GRIDS.items():
        pipe = Pipeline([("sc", StandardScaler()),
                         ("sel", SelectKBest(f_regression)), ("m", est)])
        g = dict(grid); g["sel__k"] = [25, 100, 400, "all"]
        gs = GridSearchCV(pipe, g, cv=cv, scoring="neg_mean_absolute_error",
                          n_jobs=-1).fit(Xtr, ytr)
        pt[k] = gs.predict(Xte)
        print(f"  {k}: test MAE {np.abs(pt[k]-yte).mean():.2f}")
    ih_ens = np.mean([pt[k] for k in pt], 0)
    ours = np.array([lk.loc[subs[i], "p"] for i in ite])

    # E: one feature family at a time, 5-fold CV MAE on the held-in 40
    fam_mae, fam_sd = [], []
    for pref, lab in FAMS:
        cols = [i for i, n in enumerate(names) if n.startswith(pref)]
        pipe = Pipeline([("sc", StandardScaler()),
                         ("sel", SelectKBest(f_regression, k=min(100, len(cols)))),
                         ("m", SVR(kernel="rbf", C=100, gamma=1e-3, epsilon=1.0))])
        sc = -cross_val_score(pipe, Xtr[:, cols], ytr, cv=cv,
                              scoring="neg_mean_absolute_error")
        fam_mae.append(sc.mean()); fam_sd.append(sc.std())
        print(f"  family {lab:14s} n={len(cols):4d}  CV MAE {sc.mean():.2f}")
    pipe = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=100)),
                     ("m", SVR(kernel="rbf", C=100, gamma=1e-3, epsilon=1.0))])
    allsc = -cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error")

    # D: paired bootstrap, ours vs each ensemble
    pub = np.load(os.path.join(RES, "kounios_public_preds.npz"))
    pub_ens = pub["ens"]
    rng = np.random.default_rng(0); n = len(yte)
    B = {}
    for tag, ens in (("inhouse", ih_ens), ("public", pub_ens)):
        dm, dr = [], []
        ae_o, ae_e = np.abs(ours - yte), np.abs(ens - yte)
        for _ in range(20000):
            b = rng.integers(0, n, n)
            dm.append(ae_e[b].mean() - ae_o[b].mean())
            dr.append(np.corrcoef(yte[b], ours[b])[0, 1] - np.corrcoef(yte[b], ens[b])[0, 1])
        B[f"dm_{tag}"] = np.array(dm); B[f"dr_{tag}"] = np.array(dr)

    d = dict(yte=yte, ours=ours, ih_ens=ih_ens, pub_ens=pub_ens,
             fam_mae=np.array(fam_mae), fam_sd=np.array(fam_sd),
             fam_lab=np.array([l for _, l in FAMS]), all_mae=allsc.mean(),
             svr=pt["SVR"], en=pt["ElasticNet"], gp=pt["GaussProc"], **B)
    np.savez(CACHE, **d)
    return d


def scat(ax, t, p, title, col, letter, chance):
    lo, hi = 15, 67
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1.0, ls="--", zorder=1)
    ax.scatter(t, p, s=34, facecolor=col, edgecolor="white", linewidth=0.7,
               alpha=0.9, zorder=3)
    a, b = np.polyfit(t, p, 1)
    xs = np.linspace(t.min(), t.max(), 50)
    ax.plot(xs, a * xs + b, color=col, lw=2.0, zorder=4)
    r, mae, sl = mets(t, p)
    ax.text(0.035, 0.965, f"r = {r:+.3f}\nMAE = {mae:.2f} y\nslope = {sl:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            color="#333333",
            bbox=dict(fc="white", ec="#DDDDDD", boxstyle="round,pad=0.42"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("chronological age (years)"); ax.set_ylabel("predicted age (years)")
    ax.text(-0.14, 1.07, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)


def main():
    d = compute()
    yte, ours = d["yte"], d["ours"]
    ih_ens, pub_ens = d["ih_ens"], d["pub_ens"]
    chance = np.abs(yte - np.median(yte)).mean()

    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.34)

    scat(fig.add_subplot(gs[0, 0]), yte, ih_ens,
         "Classical ensemble, trained in-house (n = 40)", RED, "A", chance)
    scat(fig.add_subplot(gs[0, 1]), yte, ours,
         "Contrastive model (t25), pretrained on 718 public", BLUE, "B", chance)

    # C: r and MAE for every model, identity by position + direct labels
    ax = fig.add_subplot(gs[0, 2])
    labs = ["SVR", "Elastic\nnet", "Gauss.\nproc.", "Ensemble", "Ours\n(t25)"]
    P = [d["svr"], d["en"], d["gp"], ih_ens, ours]
    rs = [mets(yte, p)[0] for p in P]; ms = [mets(yte, p)[1] for p in P]
    x = np.arange(5)
    cols = [RED] * 4 + [BLUE]
    ax.bar(x, ms, 0.62, color=cols, edgecolor="white", linewidth=1.4, zorder=3)
    ax.axhline(chance, color=GREY, lw=1.6, ls="--", zorder=4)
    ax.text(4.46, chance + 0.12, f"chance {chance:.2f} y", fontsize=8.6,
            color=GREY, ha="right", va="bottom")
    for xi, (m, r) in enumerate(zip(ms, rs)):
        ax.text(xi, m - 0.42, f"{m:.2f}", ha="center", va="top", fontsize=9,
                color="white", fontweight="bold")
        ax.text(xi, m - 1.30, f"r={r:+.2f}", ha="center", va="top", fontsize=8.6,
                color="white")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("MAE on held-out 41 (years)"); ax.set_ylim(0, 11.4)
    ax.text(-0.14, 1.07, "C", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    # D: bootstrap deltas -- CIs straddle zero
    ax = fig.add_subplot(gs[1, 0])
    for arr, col, lab in ((d["dm_inhouse"], RED, "vs in-house-trained"),
                          (d["dm_public"], TEAL, "vs public-trained")):
        ax.hist(arr, bins=70, color=col, alpha=0.45, density=True, label=lab)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ax.plot([lo, hi], [-0.012, -0.012], color=col, lw=3.0,
                solid_capstyle="butt", clip_on=False)
    ax.axvline(0, color="#333333", lw=1.6)
    ax.set_xlabel("$\\Delta$MAE, classical $-$ ours (years)")
    ax.set_ylabel("bootstrap density")
    ax.legend(fontsize=8.6, frameon=False, loc="upper right")
    ax.text(-0.14, 1.07, "D", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    # E: feature families alone
    ax = fig.add_subplot(gs[1, 1])
    lab = [str(l) for l in d["fam_lab"]]; mae = d["fam_mae"]; sd = d["fam_sd"]
    o = np.argsort(mae); x = np.arange(len(lab))
    ax.bar(x, mae[o], 0.62, yerr=sd[o], color=GOLD, edgecolor="white",
           linewidth=1.4, error_kw=dict(ecolor="#555555", lw=1.1, capsize=3), zorder=3)
    ax.axhline(float(d["all_mae"]), color=BLUE, lw=1.8, ls="--", zorder=4)
    ax.text(len(lab) - 0.5, float(d["all_mae"]) - 0.09, f"all families {float(d['all_mae']):.2f}",
            fontsize=8.6, color=BLUE, ha="right", va="top")
    ax.set_xticks(x); ax.set_xticklabels([lab[i] for i in o], fontsize=8.6, rotation=20,
                                         ha="right")
    ax.set_ylabel("5-fold CV MAE on the 40 held-in (years)")
    ax.text(-0.14, 1.07, "E", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    # F: the transfer panel -- public validation vs in-house hold-out
    ax = fig.add_subplot(gs[1, 2])
    pub = pd.read_csv(os.path.join(RES, "Table_kounios_public_trained.csv"))
    names = ["SVR-RBF", "ElasticNet", "GaussProc-RBF", "ensemble (mean of 3)"]
    pv = [float(pub[pub.model == n].pub_val_r.iloc[0]) if not pd.isna(
            pub[pub.model == n].pub_val_r.iloc[0]) else np.nan for n in names]
    ih = [float(pub[pub.model == n].ih_r.iloc[0]) for n in names]
    pv[-1] = np.nanmean(pv[:-1])
    x = np.arange(4); w = 0.37
    ax.bar(x - w / 2, pv, w, color=TEAL, edgecolor="white", linewidth=1.3,
           label="public validation (179)", zorder=3)
    ax.bar(x + w / 2, ih, w, color=RED, edgecolor="white", linewidth=1.3,
           label="in-house hold-out (41)", zorder=3)
    ax.axhline(0.754, color=TEAL, lw=1.5, ls=":", zorder=4)
    ax.axhline(0.537, color=BLUE, lw=1.8, ls="--", zorder=4)
    ax.text(3.46, 0.537 + 0.015, "ours, in-house 0.537", fontsize=8.4, color=BLUE,
            ha="right", va="bottom")
    ax.text(3.46, 0.754 + 0.015, "ours, public 0.754", fontsize=8.4, color=TEAL,
            ha="right", va="bottom")
    ax.axhline(0, color="#333333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["SVR", "Elastic\nnet", "Gauss.\nproc.", "Ensemble"], fontsize=9)
    ax.set_ylabel("correlation with age (r)"); ax.set_ylim(-0.12, 0.88)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left")
    ax.text(-0.14, 1.07, "F", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.18, lw=0.6); ax.set_axisbelow(True)

    for ext, kw in (("pdf", {}), ("png", dict(dpi=165))):
        fig.savefig(os.path.join(RES, f"SuppFig_classical_baseline.{ext}"),
                    bbox_inches="tight", facecolor="white", **kw)
    print("\nwrote results/SuppFig_classical_baseline.pdf (+ .png)")
    print(f"  A classical in-house  r {mets(yte,ih_ens)[0]:+.3f} MAE {mets(yte,ih_ens)[1]:.2f}")
    print(f"  B ours                r {mets(yte,ours)[0]:+.3f} MAE {mets(yte,ours)[1]:.2f}")
    print(f"  D dMAE CI inhouse [{np.percentile(d['dm_inhouse'],2.5):+.2f},{np.percentile(d['dm_inhouse'],97.5):+.2f}]"
          f"  public [{np.percentile(d['dm_public'],2.5):+.2f},{np.percentile(d['dm_public'],97.5):+.2f}]")


if __name__ == "__main__":
    main()
