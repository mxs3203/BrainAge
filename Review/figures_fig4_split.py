"""Figure 4 and its supplement: age information, and the age-model search."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from figures_fig4_information import (  # noqa: E402
    compute, probe, metrics, N_PERM, NAVY, RED, GRAY, LGRAY, RES)
from figures_fig5_agesearch import (  # noqa: E402
    load as load_trials, importances, expected_max,
    HEAD_C, STUDY_C, FLOOR_PUB, FLOOR_IH)
import channel_encoder_search_arch as A  # noqa: E402
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

# THE SHIPPED MODEL, RE-EVALUATED. Every point in the search panels was scored
# before additional phase_15 recordings were added, so those panels describe the
SHIPPED = {"study": "phase4", "trial": 25,
           "pub_r": 0.7542, "pub_mae": 8.0933,
           "ih_r_search": 0.4299, "ih_mae_search": 8.0289,
           "ih_r_now": 0.5371, "ih_mae_now": 7.4397}


PERM_CACHE = os.path.join(RES, "fig4_perm_cache.npz")


def encoder_stats():
    """Trained vs random-init vs permutation null, per channel."""
    if os.path.exists(PERM_CACHE):
        z = np.load(PERM_CACHE, allow_pickle=True)
        print(f"loaded cached permutation nulls from {PERM_CACHE}")
        return (pd.DataFrame(z["df"].item()), z["nulls"].item(),
                z["preds"].item())
    data = compute()
    rows, nulls, preds = [], {}, {}
    rng = np.random.default_rng(0)
    for ch, dd in data.items():
        t = dd["trained"]
        p = probe(t["Mtr"], t["atr"], t["Mva"], t["ava"])
        mt = metrics(t["ava"], p)
        preds[ch] = (t["ava"], p)
        q = dd["random"]
        mr = metrics(q["ava"], probe(q["Mtr"], q["atr"], q["Mva"], q["ava"]))
        nr, nm = [], []
        for _ in range(N_PERM):
            sh = rng.permutation(t["atr"])
            pp = probe(t["Mtr"], sh, t["Mva"], t["ava"])
            nr.append(np.corrcoef(t["ava"], pp)[0, 1])
            nm.append(np.abs(pp - t["ava"]).mean())
        nr, nm = np.array(nr), np.array(nm)
        nulls[ch] = {"r": nr, "mae": nm}
        rows.append({
            "ch": ch, "r": mt["r"], "mae": mt["mae"], "slope": mt["slope"],
            "r_random_init": mr["r"], "mae_random_init": mr["mae"],
            "null_r_mean": nr.mean(), "null_r_p95": np.percentile(nr, 95),
            "null_mae_mean": nm.mean(), "null_mae_p05": np.percentile(nm, 5),
            "z_r": (mt["r"] - nr.mean()) / nr.std(),
            "p_perm": (1 + (nr >= mt["r"]).sum()) / (1 + N_PERM)})
    df = pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)
    np.savez_compressed(PERM_CACHE, df=df.to_dict("list"), nulls=nulls,
                        preds=preds)
    print(f"cached permutation nulls -> {PERM_CACHE}")
    return df, nulls, preds


def search_scores():
    st = ("sqlite:///" + os.path.join(HERE, "strengthen",
                                      "channel_encoder_arch_optuna.db")
          + "?timeout=120")
    out = {}
    for ch in A.CHANNELS:
        try:
            s = optuna.load_study(study_name=f"ch_{ch}", storage=st)
        except KeyError:
            continue
        c = [t for t in s.trials if t.state.name == "COMPLETE" and t.value is not None]
        if c:
            out[ch] = -min(c, key=lambda t: t.value).value
    return out


def main():
    df, nulls, preds = encoder_stats()
    print("encoder stats done, now the age search")
    d = load_trials()
    score = search_scores()
    rng = np.random.default_rng(0)
    x = np.arange(len(df))
    best_ch = df.ch.iloc[0]
    print(f"{len(df)} channels | {len(d)} age-model trials over "
          f"{d.study.nunique()} studies")

    # ======================= MAIN FIGURE =============================
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.5))
    ax = axes.ravel()

    # ---- A  age corelation vs both nulls ---------------------------
    a = ax[0]
    a.bar(x, df.r, color=NAVY, edgecolor="k", lw=0.7, label="Trained encoder")
    a.bar(x, df.r_random_init, width=0.45, color=LGRAY, edgecolor="k", lw=0.6,
          label="Random-init, same architecture")
    a.plot(x, df.null_r_p95, ls="--", lw=1.4, color=RED,
           label="Permutation null, 95th pct")
    a.set_xticks(x); a.set_xticklabels(df.ch, rotation=45, ha="right",
                                       style="italic", fontsize=8)
    a.set_ylabel("Subject-level age correlation  r")
    a.set_ylim(min(0, df.r_random_init.min() - 0.05), df.r.max() * 1.25)
    a.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
    a.set_title("A. Age information vs chance")

    # ---- B  subject MAE vs floor and null ---------------------------
    a = ax[1]
    a.bar(x, df.mae, color=NAVY, edgecolor="k", lw=0.7, label="Trained encoder")
    a.bar(x, df.mae_random_init, width=0.45, color=LGRAY, edgecolor="k", lw=0.6,
          label="Random-init")
    a.axhline(12.883, color=RED, ls="--", lw=1.5,
              label="No-information floor (public val)")
    a.plot(x, df.null_mae_p05, ls=":", lw=1.5, color=RED,
           label="Permutation null, 5th pct")
    a.set_xticks(x); a.set_xticklabels(df.ch, rotation=45, ha="right",
                                       style="italic", fontsize=8)
    a.set_ylabel("Subject MAE (years)")
    a.legend(fontsize=6.8, loc="lower right", framealpha=0.9)
    a.set_title("B. Prediction error vs chance")

    # ---- C  effect size per channel ---------------------------------
    a = ax[2]
    a.bar(x, df.z_r, color=NAVY, edgecolor="k", lw=0.7)
    a.axhline(1.96, color=RED, ls="--", lw=1.4)
    a.text(0.0, 2.05, "p = 0.05", fontsize=7.5, color=RED, ha="left",
           va="bottom", fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(df.ch, rotation=45, ha="right",
                                       style="italic", fontsize=8)
    a.set_xlabel("Channel")
    a.set_ylabel("Effect size  (r - null mean) / null SD")
    a.set_title("C. Distance from chance, in null SDs")

    # ---- D  predicted vs true, best channel -------------------------
    a = ax[3]
    ava, p = preds[best_ch]
    a.scatter(ava, p, s=22, color=NAVY, alpha=0.6, edgecolor="none")
    lo, hi = min(ava.min(), p.min()) - 2, max(ava.max(), p.max()) + 2
    a.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1.5, label="Identity")
    b = np.polyfit(ava, p, 1)
    a.plot([lo, hi], np.polyval(b, [lo, hi]), color="k", lw=1.5,
           label=f"Fit (slope {b[0]:.2f})")
    a.axhline(np.median(ava), color=GRAY, ls=":", lw=1.2,
              label="Median-age guess")
    a.set_xlabel("Chronological age (years)")
    a.set_ylabel("Predicted age (years)")
    a.legend(fontsize=7, loc="upper left", framealpha=0.9)
    a.set_title(f"D. Predicted vs true age ({best_ch}, held-out subjects)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig4_information_transfer.{ext}"),
                    dpi=180, bbox_inches="tight")
    print("saved", os.path.join(RES, "Fig4_information_transfer.pdf"))

    # ======================= SUPPLEMENT ==============================
    f2, axes2 = plt.subplots(1, 3, figsize=(17.4, 5.6))
    s = axes2.ravel()

    # ---- A  hyperparameter importance -------------------------------
    names, imp = importances(d)
    a = s[0]
    order = np.argsort(imp["hold-out r"]["imp"])
    y = np.arange(len(names)); w = .4
    for i, (lab, c) in enumerate((("objective score", NAVY),
                                  ("hold-out r", RED))):
        a.barh(y + (i - .5) * w, imp[lab]["imp"][order], height=w,
               xerr=imp[lab]["sd"][order], color=c, alpha=.85,
               error_kw=dict(elinewidth=.7, ecolor="#333333"),
               label=f"{lab}  (OOB R2={imp[lab]['oob']:+.2f})")
    a.set_yticks(y); a.set_yticklabels([names[i] for i in order], fontsize=7.5)
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("permutation importance (drop in R2)")
    a.set_title("A. Hyperparameter importance (random forest)")
    a.legend(fontsize=7, loc="lower right")

    # ---- B  head x encoder adaptation -------------------------------
    a = s[1]
    heads = [h for h in d["head"].value_counts().index]
    xs = np.arange(len(heads)); w = .38
    for i, (lab, c) in enumerate(zip(["frozen", "fine-tuned"], (NAVY, RED))):
        m = d.freeze_encoders == (lab == "frozen")
        vals = [d[m & (d["head"] == h)].test_r.max()
                if (m & (d["head"] == h)).any() else np.nan for h in heads]
        cnt = [int((m & (d["head"] == h)).sum()) for h in heads]
        a.bar(xs + (i - .5) * w, vals, width=w, color=c, alpha=.85, label=lab)
        for x_, v_, n_ in zip(xs + (i - .5) * w, vals, cnt):
            if np.isfinite(v_):
                a.text(x_, v_ + .006, f"n={n_}", ha="center", fontsize=6.5)
    a.set_xticks(xs); a.set_xticklabels(heads, rotation=20, ha="right",
                                        style="italic")
    a.set_ylabel("best in-house hold-out r")
    a.set_title("B. Best hold-out r by head and encoder adaptation")
    a.legend(fontsize=7.5)

    # ---- C  hold-out r by head --------------------------------------
    a = s[2]
    hs = [h for h in d["head"].value_counts().index]
    dat = [d[d["head"] == h].test_r.dropna().to_numpy() for h in hs]
    bp = a.boxplot(dat, tick_labels=[f"{h}\nn={len(z)}" for h, z in zip(hs, dat)],
                   patch_artist=True, widths=.6,
                   medianprops=dict(color="k", lw=1.4))
    for patch, h in zip(bp["boxes"], hs):
        patch.set_facecolor(HEAD_C.get(h, GRAY)); patch.set_alpha(.6)
    for i, z in enumerate(dat):
        a.scatter(np.full(len(z), i + 1) + rng.normal(0, .05, len(z)), z,
                  s=9, color="k", alpha=.35, zorder=3)
    a.axhline(0, color="k", lw=1)
    a.axhline(d.test_r.mean(), color=RED, ls="--", lw=1.2,
              label=f"pooled mean {d.test_r.mean():+.3f}")
    a.set_ylabel("in-house hold-out r")
    a.set_title("C. Hold-out r by head")
    a.legend(fontsize=7.5)
    a.tick_params(axis="x", labelsize=7.5)

    f2.tight_layout()
    for ext in ("pdf", "png"):
        f2.savefig(os.path.join(RES, f"Fig4_supplement_search.{ext}"),
                   dpi=165, bbox_inches="tight")
    print("saved", os.path.join(RES, "Fig4_supplement_search.pdf"))

    # ---- numbers quoted in teh Results text -------------------------
    print("\nvalues quoted in the Results text:")
    print(f"  trained r      {df.r.min():.3f}-{df.r.max():.3f} (mean {df.r.mean():.3f})")
    print(f"  random-init r  {df.r_random_init.min():.3f}-{df.r_random_init.max():.3f} "
          f"(mean {df.r_random_init.mean():.3f})")
    print(f"  training gain  {df.r.mean() - df.r_random_init.mean():+.3f}")
    print(f"  trained MAE    {df.mae.min():.2f}-{df.mae.max():.2f} "
          f"(mean {df.mae.mean():.2f}); random-init mean {df.mae_random_init.mean():.2f}")
    print(f"  effect size z  {df.z_r.min():.2f}-{df.z_r.max():.2f}; "
          f"{(df.z_r > 1.96).sum()} of {len(df)} above 1.96")
    print(f"  max p_perm     {df.p_perm.max():.4f}")
    print(f"  beat random-init: {(df.r > df.r_random_init).sum()} of {len(df)}")
    print(f"  trials {len(d)} | hold-out r mean {d.test_r.mean():+.3f} "
          f"sd {d.test_r.std(ddof=1):.3f} max {d.test_r.max():+.3f}")
    print(f"  trials beating in-house chance ({FLOOR_IH}): "
          f"{(d.test_mae < FLOOR_IH).sum()} of {d.test_mae.notna().sum()}")
    df.round(4).to_csv(os.path.join(RES, "Table_information_vs_chance.csv"),
                       index=False)


if __name__ == "__main__":
    main()
