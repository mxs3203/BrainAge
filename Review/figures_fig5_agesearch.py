"""Figure 5: hyperparameter exploration of the AGE model, in the Figure 3 style"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.lines import Line2D

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DB = "sqlite:///" + os.path.join(HERE, "strengthen", "age_model_optuna.db")
os.makedirs(RES, exist_ok=True)

STUDIES = ["age_joint4", "age_phase2c", "age_phase3", "age_phase4",
           "age_phase5", "age_phase6"]
# teh hyperparameters every one of the six studies searched
CORE = ["head", "ih_oversample", "age_strat", "aux", "freeze_encoders",
        "dropout", "head_hidden", "lr_head", "lr_encoder", "weight_decay",
        "loss", "bag_passes", "bag_k", "bag_subjects"]
CAT = ["head", "aux", "loss", "age_strat", "freeze_encoders"]
# chance floors (subject-level MAD-from-median MAE), from the trace of t25
FLOOR_PUB, FLOOR_IH = 12.883, 9.098

NAVY, RED, GRAY = "#1F4E79", "#C00000", "#808080"
HEAD_C = {"logeuclid": NAVY, "concat": "#2E7D5B", "attention": RED,
          "transformer": "#C98B2E", "gated": "#6A3D9A"}
STUDY_C = {"joint4": "#1F4E79", "phase2c": "#2E7D5B", "phase3": "#C98B2E",
           "phase4": "#C00000", "phase5": "#6A3D9A", "phase6": "#17A2B8"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load():
    rows = []
    for s in STUDIES:
        st = optuna.load_study(study_name=s, storage=DB)
        for t in st.trials:
            if t.state.name != "COMPLETE" or not all(k in t.params for k in CORE):
                continue
            a = t.user_attrs
            r = {"study": s.replace("age_", ""), "trial": t.number,
                 "score": t.value}
            r.update({k: t.params[k] for k in CORE})
            for k in ("r", "subject_mae", "slope", "test_r", "test_mae",
                      "test_slope", "inhouse_r", "inhouse_mae", "epoch",
                      "cross_cohort_r", "feasible", "beats_floor",
                      "locked_in_objective"):
                r[k] = a.get(k, np.nan)
            r["cost"] = (t.params["bag_passes"] * t.params["bag_k"]
                         * t.params["ih_oversample"]
                         * (32.0 / max(1, t.params["bag_subjects"])))
            rows.append(r)
    d = pd.DataFrame(rows)
    d["locked_in_objective"] = d["locked_in_objective"].fillna(False).infer_objects(copy=False).astype(bool)
    return d


def design_matrix(d):
    """Numeric design matrix over CORE, with categoricals ordinal-coded."""
    X, names = [], []
    for c in CORE:
        v = d[c]
        if c in CAT or v.dtype == object or v.dtype == bool:
            codes = pd.Categorical(v.astype(str)).codes.astype(float)
            X.append(codes)
        elif c in ("lr_head", "lr_encoder", "weight_decay"):
            X.append(np.log10(v.astype(float)))
            c = f"log10 {c}"
        else:
            X.append(v.astype(float))
        names.append(c)
    return np.column_stack(X), names


def importances(d):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    X, names = design_matrix(d)
    out = {}
    for tgt, lab in (("score", "objective score"), ("test_r", "validation r")):
        y = d[tgt].to_numpy(float)
        k = np.isfinite(y) & np.isfinite(X).all(1)
        rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                                   oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(X[k], y[k])
        pi = permutation_importance(rf, X[k], y[k], n_repeats=30,
                                    random_state=0, n_jobs=-1)
        out[lab] = dict(imp=pi.importances_mean, sd=pi.importances_std,
                        oob=float(rf.oob_score_), n=int(k.sum()))
    return names, out


def expected_max(n, mu, sd, rng, reps=4000):
    """E[max of n draws from N(mu, sd)], by simulation."""
    return float(np.mean(rng.normal(mu, sd, size=(reps, n)).max(1)))


def main():
    d = load()
    d.to_csv(os.path.join(RES, "Table_age_hparam_trials.csv"), index=False)
    print(f"{len(d)} complete trials over {d.study.nunique()} studies")
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(4, 3, figsize=(19, 21))
    ax = axes.ravel()

    # ---- A  search progress ------------------------------------------
    a = ax[0]
    off = 0
    for s, g in d.groupby("study", sort=False):
        g = g.sort_values("trial")
        x = np.arange(len(g)) + off
        c = STUDY_C.get(s, GRAY)
        a.plot(x, g.score, "o", ms=2.6, color=c, alpha=.35)
        a.plot(x, g.score.cummin(), "-", lw=1.8, color=c,
               label=f"{s} (n={len(g)}, best {g.score.min():.3f})")
        a.axvline(off - .5, color="k", lw=.6, alpha=.35)
        a.text(off + len(g) / 2, a.get_ylim()[1], s, ha="center", va="top",
               fontsize=7, style="italic", color=GRAY)
        off += len(g)
    a.set_xlabel("trial (studies concatenated)")
    a.set_ylabel("objective score (lower is better)")
    a.set_title("A. Search progress, running best per study")
    a.legend(fontsize=7, ncol=2)

    # ---- B  hyperparameter importance --------------------------------
    names, imp = importances(d)
    a = ax[1]
    order = np.argsort(imp["validation r"]["imp"])
    y = np.arange(len(names))
    w = .4
    for i, (lab, c) in enumerate((("objective score", NAVY),
                                  ("validation r", RED))):
        a.barh(y + (i - .5) * w, imp[lab]["imp"][order], height=w,
               xerr=imp[lab]["sd"][order], color=c, alpha=.85,
               error_kw=dict(elinewidth=.7, ecolor="#333333"),
               label=f"{lab}  (OOB R²={imp[lab]['oob']:+.2f})")
    a.set_yticks(y); a.set_yticklabels([names[i] for i in order], fontsize=7.5)
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("permutation importance (drop in R²)")
    a.set_title("B. Hyperparameter importance (random forest)")
    a.legend(fontsize=7, loc="lower right")
    if max(imp[l]["oob"] for l in imp) < 0.1:
        a.text(.5, .04, "both OOB R² ≈ 0: hyperparameters do not\n"
                        "explain the outcome at this sample size",
               transform=a.transAxes, ha="center", fontsize=7.5,
               color=RED, style="italic")

    # ---- C  PCA biplot ------------------------------------------------
    a = ax[2]
    X, nm = design_matrix(d)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc = U[:, :2] * S[:2]
    var = S ** 2 / (S ** 2).sum()
    sc = a.scatter(pc[:, 0], pc[:, 1], c=d.test_r, cmap="RdYlBu_r", s=26,
                   edgecolor="w", lw=.4, vmin=-.1, vmax=.45)
    fig.colorbar(sc, ax=a, fraction=.046, label="in-house hold-out r")
    k = 3.2
    for i, n_ in enumerate(nm):
        if np.hypot(*Vt[:2, i]) < .18:
            continue
        a.arrow(0, 0, Vt[0, i] * k, Vt[1, i] * k, color=RED, lw=1,
                head_width=.09, length_includes_head=True)
        a.text(Vt[0, i] * k * 1.13, Vt[1, i] * k * 1.13, n_, color=RED,
               fontsize=6.5, ha="center", fontweight="bold")
    a.axhline(0, color=GRAY, ls="--", lw=.8); a.axvline(0, color=GRAY, ls="--", lw=.8)
    a.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    a.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    a.set_title("C. PCA biplot of trial hyperparameter profiles")

    # ---- D  categorical choices ---------------------------------------
    a = ax[3]
    heads = [h for h in d["head"].value_counts().index]
    adapt = ["frozen", "fine-tuned"]
    w = .38
    xs = np.arange(len(heads))
    for i, (lab, c) in enumerate(zip(adapt, (NAVY, RED))):
        m = d.freeze_encoders == (lab == "frozen")
        vals = [d[m & (d["head"] == h)].test_r.max() if (m & (d["head"] == h)).any()
                else np.nan for h in heads]
        cnt = [int((m & (d["head"] == h)).sum()) for h in heads]
        a.bar(xs + (i - .5) * w, vals, width=w, color=c, alpha=.85, label=lab)
        for x_, v_, n_ in zip(xs + (i - .5) * w, vals, cnt):
            if np.isfinite(v_):
                a.text(x_, v_ + .006, f"n={n_}", ha="center", fontsize=6.5)
    a.set_xticks(xs); a.set_xticklabels(heads, rotation=20, ha="right",
                                        style="italic")
    a.set_ylabel("best in-house hold-out r")
    a.set_title("D. Best hold-out r by head and encoder adaptation")
    a.legend(fontsize=7.5)

    # ---- E  learning-rate plane ---------------------------------------
    a = ax[4]
    sc = a.scatter(d.lr_encoder, d.lr_head, c=d.test_r, cmap="RdYlBu_r", s=30,
                   edgecolor="w", lw=.4, vmin=-.1, vmax=.45)
    best = d.loc[d.test_r.idxmax()]
    a.scatter(best.lr_encoder, best.lr_head, marker="*", s=320, color="none",
              edgecolor="k", lw=1.4, zorder=5)
    a.annotate(f"best  r={best.test_r:+.3f}",
               (best.lr_encoder, best.lr_head), textcoords="offset points",
               xytext=(8, 8), fontsize=7.5, fontweight="bold")
    a.set_xscale("log"); a.set_yscale("log")
    fig.colorbar(sc, ax=a, fraction=.046, label="hold-out r")
    a.set_xlabel("lr_encoder"); a.set_ylabel("lr_head")
    a.set_title("E. Learning-rate plane")

    # ---- F  regularisation plane --------------------------------------
    a = ax[5]
    sc = a.scatter(d.dropout, d.weight_decay, c=d.test_r, cmap="RdYlBu_r",
                   s=30, edgecolor="w", lw=.4, vmin=-.1, vmax=.45)
    a.scatter(best.dropout, best.weight_decay, marker="*", s=320, color="none",
              edgecolor="k", lw=1.4, zorder=5)
    a.set_yscale("log")
    fig.colorbar(sc, ax=a, fraction=.046, label="hold-out r")
    a.set_xlabel("dropout"); a.set_ylabel("weight decay")
    a.set_title("F. Regularisation plane")

    # ---- G  public r vs hold-out r ------------------------------------
    a = ax[6]
    szs = 14 + 110 * (d.cost - d.cost.min()) / (d.cost.max() - d.cost.min())
    for h, g in d.groupby("head"):
        m = d["head"] == h
        a.scatter(g.r, g.test_r, s=szs[m], alpha=.65,
                  color=HEAD_C.get(h, GRAY), edgecolor="w", lw=.4, label=h)
    a.axhline(0, color="k", lw=1)
    a.axvline(0.72, color=RED, ls="--", lw=1)
    a.text(0.722, a.get_ylim()[0] + .02, "public feasibility r≥0.72",
           color=RED, fontsize=6.5, rotation=90, va="bottom")
    a.set_xlabel("public hold-out r (179 subjects)")
    a.set_ylabel("in-house hold-out r (41 subjects)")
    a.set_title("G. Transfer: public vs in-house (bubble = compute cost)")
    a.legend(fontsize=7, title="head", title_fontsize=7)

    # ---- H  MAE plane with floors and Pareto front ---------------------
    a = ax[7]
    a.scatter(d.subject_mae, d.test_mae, s=24, color=NAVY, alpha=.55,
              edgecolor="w", lw=.4)
    a.axvline(FLOOR_PUB, color=RED, ls="-.", lw=1.2)
    a.axhline(FLOOR_IH, color=RED, ls="-.", lw=1.2)
    a.text(FLOOR_PUB - .1, a.get_ylim()[0], "public chance", color=RED,
           fontsize=6.5, rotation=90, va="bottom", ha="right")
    a.text(a.get_xlim()[0], FLOOR_IH + .05, "in-house chance", color=RED,
           fontsize=6.5)
    p = d.dropna(subset=["subject_mae", "test_mae"]).sort_values("subject_mae")
    front, bestm = [], np.inf
    for _, rw in p.iterrows():
        if rw.test_mae < bestm:
            bestm = rw.test_mae; front.append(rw)
    pf = pd.DataFrame(front)
    a.plot(pf.subject_mae, pf.test_mae, "-o", color=RED, ms=4, lw=1.4,
           label=f"Pareto front (n={len(pf)})")
    lab_n = 0
    for _, rw in pf.iterrows():
        if rw.test_mae < 8.6:
            lab_n += 1
            a.annotate(f"{rw.study} t{int(rw.trial)}",
                       (rw.subject_mae, rw.test_mae),
                       textcoords="offset points",
                       xytext=(8, -4 + 11 * (lab_n % 2)), fontsize=6.5,
                       fontweight="bold",
                       arrowprops=dict(arrowstyle="-", lw=.5, color=GRAY))
    a.set_xlabel("public MAE (years)"); a.set_ylabel("in-house hold-out MAE (years)")
    a.set_title("H. Error plane, with chance floors")
    a.legend(fontsize=7.5)

    # ---- I  validation vs hold-out ------------------------------------
    # Pooled over all 214 trials this looks like a weak positive association,
    a = ax[8]
    per = []
    for st_, g in d.groupby("study", sort=False):
        kk = g.dropna(subset=["inhouse_r", "test_r"])
        if len(kk) < 5:
            continue
        rr_ = np.corrcoef(kk.inhouse_r, kk.test_r)[0, 1]
        per.append((st_, len(kk), rr_))
        a.scatter(kk.inhouse_r, kk.test_r, s=24, alpha=.7,
                  color=STUDY_C.get(st_, GRAY), edgecolor="w", lw=.4,
                  label=f"{st_}  r={rr_:+.3f}")
    k = d.dropna(subset=["inhouse_r", "test_r"])
    rr = np.corrcoef(k.inhouse_r, k.test_r)[0, 1]
    b_ = np.polyfit(k.inhouse_r, k.test_r, 1)
    xs = np.linspace(k.inhouse_r.min(), k.inhouse_r.max(), 40)
    a.plot(xs, np.polyval(b_, xs), color="k", lw=1.8,
           label=f"pooled  r={rr:+.3f}")
    a.axhline(0, color=GRAY, lw=.8); a.axvline(0, color=GRAY, lw=.8)
    a.set_xlabel("in-house VALIDATION r (20 subjects)")
    a.set_ylabel("in-house HOLD-OUT r (41 subjects)")
    lo, hi = min(x[2] for x in per), max(x[2] for x in per)
    a.set_title(f"I. Does validation rank configs?  per-study r "
                f"{lo:+.2f} to {hi:+.2f}")
    a.legend(fontsize=6.8, loc="upper left", ncol=2)
    a.text(.97, .04, "the sign flips between studies: the 20-subject\n"
                     "validation split does not rank configurations",
           transform=a.transAxes, fontsize=7.5, color=RED, style="italic",
           ha="right")

    # ---- J  best-of-K order statistics --------------------------------
    a = ax[9]
    v = d.test_r.dropna().to_numpy()
    mu, sd = v.mean(), v.std(ddof=1)
    ns = np.arange(1, len(v) + 1)
    em = [expected_max(int(n), mu, sd, rng) for n in ns]
    a.plot(ns, em, color=NAVY, lw=1.8,
           label=f"E[max of n]  if r ~ N({mu:+.3f}, {sd:.3f})")
    run = pd.Series(v).cummax()
    a.plot(ns, run, color=RED, lw=1.5, label="observed running max")
    a.axhline(v.max(), color=GRAY, ls="--", lw=1)
    a.text(.97, .55, f"observed max {v.max():+.3f}\n"
                     f"expected {em[-1]:+.3f} from {len(v)} draws\n"
                     f"-> the winner is what {len(v)} draws produce",
           transform=a.transAxes, ha="right", fontsize=8, fontweight="bold",
           bbox=dict(fc="w", ec=GRAY, alpha=.85, boxstyle="round,pad=0.35"))
    a.set_xlabel("number of trials n"); a.set_ylabel("in-house hold-out r")
    a.set_title("J. Best-of-K: is the winner more than the best of 214?")
    a.legend(fontsize=7.5, loc="lower right")

    # ---- K  hold-out r by head ----------------------------------------
    a = ax[10]
    hs = [h for h in d["head"].value_counts().index]
    data = [d[d["head"] == h].test_r.dropna().to_numpy() for h in hs]
    bp = a.boxplot(data, tick_labels=[f"{h}\nn={len(x)}" for h, x in zip(hs, data)],
                   patch_artist=True, widths=.6,
                   medianprops=dict(color="k", lw=1.4))
    for patch, h in zip(bp["boxes"], hs):
        patch.set_facecolor(HEAD_C.get(h, GRAY)); patch.set_alpha(.6)
    for i, x in enumerate(data):
        a.scatter(np.full(len(x), i + 1) + rng.normal(0, .05, len(x)), x,
                  s=9, color="k", alpha=.35, zorder=3)
    a.axhline(0, color="k", lw=1)
    a.axhline(d.test_r.mean(), color=RED, ls="--", lw=1.2,
              label=f"pooled mean {d.test_r.mean():+.3f}")
    a.set_ylabel("in-house hold-out r")
    a.set_title("K. Hold-out r by head")
    a.legend(fontsize=7.5)
    a.tick_params(axis="x", labelsize=7.5)

    # ---- L  cost vs performance ---------------------------------------
    a = ax[11]
    for h, g in d.groupby("head"):
        a.scatter(g.cost, g.test_r, s=26, alpha=.7, color=HEAD_C.get(h, GRAY),
                  edgecolor="w", lw=.4, label=h)
    a.set_xscale("log")
    a.axvline(4000, color=RED, ls="--", lw=1.2)
    a.text(4200, a.get_ylim()[0] + .02, "prune guard", color=RED, fontsize=6.5,
           rotation=90, va="bottom")
    cb = d.dropna(subset=["cost", "test_r"])
    rc = np.corrcoef(np.log10(cb.cost), cb.test_r)[0, 1]
    a.set_xlabel("compute cost  (passes × bag_k × oversample × 32/subjects)")
    a.set_ylabel("in-house hold-out r")
    a.set_title(f"L. Does more compute buy accuracy?  r = {rc:+.3f}")
    a.legend(fontsize=7, title="head", title_fontsize=7)

    fig.suptitle("Figure 5. Hyperparameter exploration of the age model: "
                 f"{len(d)} trials over {d.study.nunique()} Optuna studies",
                 fontsize=15, fontweight="bold", y=.998)
    fig.tight_layout(rect=[0, 0, 1, .985])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig5_age_hparam_search.{ext}"),
                    dpi=160, bbox_inches="tight")

    it = pd.DataFrame({"hyperparameter": names})
    for lab in imp:
        it[f"imp_{lab.replace(' ', '_')}"] = imp[lab]["imp"]
        it[f"sd_{lab.replace(' ', '_')}"] = imp[lab]["sd"]
        it[f"oob_r2_{lab.replace(' ', '_')}"] = imp[lab]["oob"]
    it.round(5).to_csv(os.path.join(RES, "Table_age_hparam_importance.csv"),
                       index=False)

    print("\nOOB R^2:", {k: round(v["oob"], 3) for k, v in imp.items()})
    print(f"val-vs-holdout r = {rr:+.3f} | cost-vs-holdout r = {rc:+.3f}")
    print(f"hold-out r: mean {mu:+.3f} sd {sd:.3f} | observed max {v.max():+.3f}"
          f" | E[max of {len(v)}] {em[-1]:+.3f}")
    print("\nwrote results/Fig5_age_hparam_search.pdf/.png")
    print("      results/Table_age_hparam_importance.csv")
    print("      results/Table_age_hparam_trials.csv")


if __name__ == "__main__":
    main()
