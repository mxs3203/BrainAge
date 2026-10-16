"""Predicted vs chronological age in four panels, for one joint-trained model."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

NAVY, RED, GRAY = "#1F4E79", "#C00000", "#808080"
AMBER, PURPLE, GREEN = "#C98B2E", "#6A3D9A", "#2E7D5B"
TEAL = "#1B7F79"


def stats(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    r = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 and t.std() > 0 else np.nan
    return dict(n=len(t), r=r, mae=float(np.abs(p - t).mean()),
                slope=float(np.polyfit(t, p - t, 1)[0]) if len(t) > 2 else np.nan,
                chance=float(np.abs(t - np.median(t)).mean()))


def _frame(ax, t, p, letter):
    """Shared axis furniture: identity line, limits, labels, panel letter."""
    lo = min(t.min(), p.min()) - 3
    hi = max(t.max(), p.max()) + 3
    ax.plot([lo, hi], [lo, hi], "--", color=RED, lw=1.5, label="identity",
            zorder=1)
    ax.set_xlabel("chronological age (years)")
    ax.set_ylabel("predicted age (years)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(alpha=.25)
    # panel letter, outside the axes so it never lands on a data point
    ax.text(-0.085, 1.045, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=17, fontweight="bold", color="black")
    return lo, hi


def _tag(ax, tag, tagcolor):
    ax.text(.97, .97, tag, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.8, fontweight="bold", color=tagcolor,
            bbox=dict(fc="white", ec=tagcolor, alpha=.85,
                      boxstyle="round,pad=0.28"))


def _margin(ax, st):
    """How far below teh median-guess baseline, as a percentage."""
    marg = 100.0 * (1.0 - st["mae"] / st["chance"])
    mcol = GREEN if marg >= 10 else (AMBER if marg > 0 else RED)
    ax.text(.03, .03, f"{marg:+.0f}% vs chance", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=mcol)
    st["pct_vs_chance"] = marg
    return st


def panel(ax, groups, title, tag, tagcolor, letter):
    """One pooled panel. groups: list of (label, frame, colour, alpha)."""
    allt, allp = [], []
    for lab, g, c, al in groups:
        ax.scatter(g.t, g.p, s=24, alpha=al, edgecolor="none", color=c,
                   label=f"{lab} (n={len(g)})", zorder=3)
        allt.append(g.t.to_numpy(float)); allp.append(g.p.to_numpy(float))
    t = np.concatenate(allt); p = np.concatenate(allp)
    st = stats(t, p)
    lo, hi = _frame(ax, t, p, letter)
    if np.isfinite(st["r"]):
        b, a0 = np.polyfit(t, p, 1)
        ax.plot([lo, hi], [a0 + b * lo, a0 + b * hi], "-", color="k", lw=1.8,
                label=f"fit (slope {b:.2f})", zorder=4)
    ax.axhline(np.median(t), ls=":", color=GRAY, lw=1.4,
               label="median-age guess", zorder=2)
    txt = (f"n = {st['n']}\nr = {st['r']:+.3f}\nMAE = {st['mae']:.2f}\n"
           f"chance MAE = {st['chance']:.2f}\nbias slope = {st['slope']:+.3f}")
    ax.text(.03, .97, txt, transform=ax.transAxes, va="top", fontsize=8.5,
            family="monospace", bbox=dict(fc="white", ec="0.7", alpha=.92))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=14)
    _tag(ax, tag, tagcolor)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=.92)
    return _margin(ax, st)


def panel_by_dataset(ax, groups, title, tag, tagcolor, letter):
    """Panel C: one colour, one fit line and one stats row per dataset."""
    allt, allp = [], []
    lines = []
    for lab, g, c in groups:
        t = g.t.to_numpy(float); p = g.p.to_numpy(float)
        ax.scatter(t, p, s=26, alpha=.75, edgecolor="none", color=c,
                   label=f"{lab} (n={len(g)})", zorder=3)
        allt.append(t); allp.append(p)
        st = stats(t, p)
        lines.append((lab, c, st))
    T = np.concatenate(allt); P = np.concatenate(allp)
    lo, hi = _frame(ax, T, P, letter)
    for (lab, c, st), (_, g, _c) in zip(lines, [(l, g, c) for l, g, c in groups]):
        t = g.t.to_numpy(float); p = g.p.to_numpy(float)
        if np.isfinite(st["r"]):
            b, a0 = np.polyfit(t, p, 1)
            # clipped to each dataset's own age range: extending a 25-subject
            # fit across the full axis would imply support that is not there
            x0, x1 = t.min(), t.max()
            ax.plot([x0, x1], [a0 + b * x0, a0 + b * x1], "-", color=c, lw=2.0,
                    zorder=4)
        ax.axhline(np.median(t), ls=":", color=c, lw=1.2, alpha=.7, zorder=2)
    box = []
    for lab, c, st in lines:
        marg = 100.0 * (1.0 - st["mae"] / st["chance"])
        st["pct_vs_chance"] = marg      # printed per dataset, so keep it in teh table
        box.append(f"{lab:<9s} n={st['n']:>3d}  r={st['r']:+.3f}  "
                   f"MAE={st['mae']:.2f}  chance={st['chance']:.2f}  "
                   f"{marg:+.0f}%")
    ax.text(.03, .97, "\n".join(box), transform=ax.transAxes, va="top",
            fontsize=7.6, family="monospace",
            bbox=dict(fc="white", ec="0.7", alpha=.92))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=14)
    _tag(ax, tag, tagcolor)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=.92)
    # no pooled margin printed -- the pooled set IS panel B, see the docstring
    return [(lab, st) for lab, c, st in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="age_phase4_t25_newdata")
    a = ap.parse_args()
    src = os.path.join(RES, f"Table_pred_joint_{a.config}.csv")
    df = pd.read_csv(src)
    print(f"read {len(df)} prediction rows from {os.path.basename(src)}")

    tr_pub = df[df.split == "public_train"]
    if not len(tr_pub):
        raise SystemExit(
            f"{src} has no public_train rows. Regenerate it with:\n"
            f"  cd strengthen && python predict_best.py --config {a.config} "
            f"--train-preds")
    fit_ih = df[df.split == "ih_fit"]
    val_pub = df[df.split == "public_val"]
    lock_ih = df[df.split == "ih_locked"]
    dsets = sorted(val_pub.cohort.unique())

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.4))
    ax = axes.ravel()
    rows = []

    rows.append(("A all training data", panel(
        ax[0],
        [("public train", tr_pub, NAVY, .45),
         ("in-house FIT", fit_ih, AMBER, .95)],
        "ALL TRAINING DATA (public pool + in-house fit)",
        "IN-SAMPLE - not a result", RED, "A")))

    rows.append(("B all validation data", panel(
        ax[1], [("public validation", val_pub, NAVY, .55)],
        "ALL VALIDATION DATA (public, pooled)",
        "held out", GREEN, "B")))

    cols = [NAVY, TEAL, PURPLE, AMBER]
    groups = [(ds.replace("public: ", ""), val_pub[val_pub.cohort == ds],
               cols[i % len(cols)]) for i, ds in enumerate(dsets)]
    for lab, st in panel_by_dataset(
            ax[2], groups,
            "PUBLIC DATASETS, split by cohort",
            "held out (panel B, split)", GREEN, "C"):
        rows.append((f"C public: {lab}", st))

    rows.append(("D in-house hold-out", panel(
        ax[3], [("in-house hold-out", lock_ih, PURPLE, .9)],
        "IN-HOUSE HOLD-OUT (41, locked)",
        "held out; config choice still selection-advantaged", PURPLE, "D")))

    fig.suptitle("Predicted against chronological age for the reported model "
                 "(joint-trained: public pool + 40 in-house participants)",
                 fontsize=13, fontweight="bold", y=.997)
    fig.tight_layout(rect=[0, 0, 1, .972])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"Fig_pred4_{a.config}.{ext}"), dpi=170,
                    bbox_inches="tight")

    sm = pd.DataFrame([dict(panel=k, **v) for k, v in rows])
    sm["beats_chance"] = sm.mae < sm.chance
    sm.round(4).to_csv(os.path.join(RES, f"Table_pred4_{a.config}.csv"),
                       index=False)
    print(f"=== {a.config} ===")
    print(sm.round(3).to_string(index=False))
    print(f"wrote results/Fig_pred4_{a.config}.pdf/.png")


if __name__ == "__main__":
    main()
