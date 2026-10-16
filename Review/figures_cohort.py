"""Main cohort figure, and the stimulus-quality supplement."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harmonized_io import load_dataset, DATASETS as ALL_DATASETS, info  # noqa: E402

BRAINAGE = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
XLSX = os.path.join(BRAINAGE, "data", "PilotDatasetSubjects.xlsx")
STIM_CSV = os.path.join(BRAINAGE, "stimuli_descriptions_and_expectations.csv")

DATASETS = [d for d in os.environ.get(
    "FIG_DATASETS", "ours,ds005385,ds003775").split(",") if d] or ALL_DATASETS

OURS, PUB = "#C62828", "#123B7A"
GRAY, GREEN = "#6E6E6E", "#2E7D32"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

LABEL = {"ours": "ours\n(EPOC X)", "ds005385": "ds005385", "ds003775": "ds003775"}
DESC = {"ours": "task-evoked + rest", "ds005385": "rest, pre/post task",
        "ds003775": "rest, eyes-closed"}


def panel(ax, letter, dx=-0.13, dy=1.08):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom")


def collect():
    D = {}
    for d in DATASETS:
        X, recs = load_dataset(d)
        per = recs.groupby("subject").agg(age=("age", "first"))
        D[d] = {"ages": per.age.to_numpy()}
    return D


def inhouse_demographics(subjects):
    """Age / sex / education for the analysed participants only."""
    d = pd.read_excel(XLSX, skiprows=1)
    d.columns = [str(c).strip() for c in d.columns]
    d = d[["ID", "Age", "Gender", "Education"]].dropna(subset=["ID"])
    d["ID"] = d["ID"].astype(str).str.strip()
    keep = d[d.ID.isin(set(subjects))].copy()
    keep["Age"] = pd.to_numeric(keep["Age"], errors="coerce")
    return keep.dropna(subset=["Age"])


def stimulus_quality(subjects):
    """Pooled kept/discarded rows per stimulus across the analysed cohort."""
    parts = []
    for fp in glob.glob(os.path.join(BRAINAGE, "results", "BA*",
                                     "2_consecutive_rows_report.csv")):
        sid = os.path.basename(os.path.dirname(fp))
        if sid not in subjects:
            continue
        df = pd.read_csv(fp)
        df["ID"] = sid
        parts.append(df)
    if not parts:
        return None
    allr = pd.concat(parts, ignore_index=True)
    g = allr.groupby("task_name").agg(total=("total_rows", "sum"),
                                      kept=("kept_rows", "sum")).reset_index()
    g["discarded_percent"] = 100.0 * (1 - g.kept / g.total.clip(lower=1))

    # label stimuli the way the original figure did: cognitive domain plus a
    # running index within that domain
    sd = pd.read_csv(STIM_CSV)
    dom = dict(zip(sd.original_stimuli_name, sd.cognitive_domain))
    g = g[g.task_name.isin(dom)].copy()
    g["domain"] = g.task_name.map(dom)
    g["label"] = ""
    for dm, idx in g.groupby("domain").groups.items():
        for k, i in enumerate(idx, start=1):
            g.loc[i, "label"] = f"{dm} task {k}"
    return g.sort_values("kept", ascending=False).reset_index(drop=True)


def main():
    D = collect()
    print(f"collected {len(D)} cohorts")
    order = ["ours"] + [d for d in DATASETS if d != "ours"]
    cols = [OURS if d == "ours" else PUB for d in order]
    _, recs_ours = load_dataset("ours")
    subjects = set(recs_ours.subject.astype(str))
    dem = inhouse_demographics(subjects)

    fig = plt.figure(figsize=(14, 14.5))
    gs = GridSpec(3, 6, figure=fig, hspace=0.62, wspace=0.95)

    # ---- A: participants per dataset ---------------------------------
    ax = fig.add_subplot(gs[0, 0:3])
    n = [len(D[d]["ages"]) for d in order]
    y = np.arange(len(order))
    ax.barh(y, n, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[d] for d in order], fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(n):
        ax.text(v + 8, i, f"{v}  ({DESC[order[i]]})", va="center", fontsize=7.5)
    ax.set_xlim(0, max(n) * 1.55)
    ax.set_xlabel("participants")
    ax.set_title(f"A. Participants per dataset (total {sum(n)})", loc="left")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "A")

    # ---- B: age distribution per dataset ------------------------------
    ax = fig.add_subplot(gs[0, 3:6])
    data = [D[d]["ages"] for d in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    tick_labels=[LABEL[d] for d in order])
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor("black")
    for el in bp["medians"]:
        el.set_color("black"); el.set_linewidth(1.6)
    rng = np.random.default_rng(0)
    for i, a in enumerate(data, start=1):
        ax.scatter(a, rng.normal(i, 0.07, len(a)), s=5, color=cols[i - 1],
                   alpha=0.45, zorder=3, linewidths=0)
    ax.invert_yaxis()
    ax.set_xlabel("age (years)")
    ax.set_title("B. Age distribution by dataset", loc="left")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "B")

    # ---- C: pooled age coverage ---------------------------------------
    ax = fig.add_subplot(gs[1, 0:3])
    ours = D["ours"]["ages"]
    pub = np.concatenate([D[d]["ages"] for d in order if d != "ours"])
    bins = np.arange(15, 80, 5)
    ax.hist(pub, bins=bins, color=PUB, alpha=0.85, edgecolor="black",
            linewidth=0.5, label=f"public (n={len(pub)})")
    ax.hist(ours, bins=bins, color=OURS, alpha=0.85, edgecolor="black",
            linewidth=0.5, label=f"ours (n={len(ours)})")
    ax.set_xlabel("age (years)"); ax.set_ylabel("participants")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C. Pooled age coverage", loc="left")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "C")

    # ---- D: coverage gain in older adults -----------------------------
    ax = fig.add_subplot(gs[1, 3:6])
    edges = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 75)]
    lo = [int(((ours >= a) & (ours < b)).sum()) for a, b in edges]
    hi = [int(((pub >= a) & (pub < b)).sum()) for a, b in edges]
    x = np.arange(len(edges)); w = 0.38
    ax.bar(x - w / 2, lo, w, color=OURS, edgecolor="black", linewidth=0.6,
           label="ours")
    ax.bar(x + w / 2, hi, w, color=PUB, edgecolor="black", linewidth=0.6,
           label="public")
    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.text(i - w / 2, a + 3, str(a), ha="center", fontsize=7.5)
        ax.text(i + w / 2, b + 3, str(b), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}-{b-1}" for a, b in edges])
    ax.set_xlabel("age band"); ax.set_ylabel("participants")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("D. Coverage gain, especially in older adults", loc="left")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "D")

    # ---- E: in-house age distribution ---------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    a = dem.Age.to_numpy(float)
    ax.hist(a, bins=np.arange(15, 70, 5), color=OURS, alpha=0.9,
            edgecolor="black", linewidth=0.5)
    ax.axvline(a.mean(), color="black", ls="--", lw=1.4,
               label=f"mean = {a.mean():.1f}")
    ax.axvline(np.median(a), color=GRAY, ls=":", lw=1.6,
               label=f"median = {np.median(a):.0f}")
    ax.set_xlabel("age (years)"); ax.set_ylabel("participants")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"E. In-house age (n={len(a)})", loc="left")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "E", dx=-0.22)

    # ---- F: education -------------------------------------------------
    ax = fig.add_subplot(gs[2, 2:4])
    ed = dem.Education.value_counts()
    ax.bar(range(len(ed)), ed.values, color=OURS, edgecolor="black",
           linewidth=0.6)
    for i, v in enumerate(ed.values):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(ed)))
    ax.set_xticklabels(ed.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("participants"); ax.set_xlabel("education")
    ax.set_ylim(0, ed.values.max() * 1.18)
    ax.set_title("F. Education level", loc="left")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "F", dx=-0.22)

    # ---- G: age by sex ------------------------------------------------
    ax = fig.add_subplot(gs[2, 4:6])
    groups, labels, ccols = [], [], []
    for g_, lab, c in (("F", "Female", "#E8A0A0"), ("M", "Male", "#9FB6D4")):
        v = dem.loc[dem.Gender.astype(str).str.upper().str[0] == g_,
                    "Age"].to_numpy(float)
        if len(v):
            groups.append(v); labels.append(f"{lab}\n(n={len(v)})"); ccols.append(c)
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55, tick_labels=labels)
    for p, c in zip(bp["boxes"], ccols):
        p.set_facecolor(c); p.set_edgecolor("black")
    for el in bp["medians"]:
        el.set_color("black"); el.set_linewidth(1.6)
    rng = np.random.default_rng(1)
    for i, v in enumerate(groups, start=1):
        ax.scatter(rng.normal(i, 0.055, len(v)), v, s=10, color="black",
                   alpha=0.45, zorder=3, linewidths=0)
    ax.set_ylabel("age (years)"); ax.set_xlabel("sex")
    ax.set_title("G. Age by sex", loc="left")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    panel(ax, "G", dx=-0.26)

    out = os.path.join(RES, "Fig_cohort.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")
    print(f"  in-house demographics over {len(dem)} analysed participants")
    print(f"  age mean {a.mean():.1f} median {np.median(a):.0f}")
    print(f"  education {dict(ed)}")

    # ================= supplement: stimulus quality ====================
    q = stimulus_quality(subjects)
    if q is None:
        print("  no per-stimulus quality reports found; supplement skipped")
        return
    f2, ax2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                           gridspec_kw={"hspace": 0.08})
    xs = np.arange(len(q))
    ax2[0].bar(xs, q.kept, color="#123B7A", edgecolor="black", linewidth=0.4)
    ax2[0].set_ylabel("kept rows")
    ax2[0].set_title("A. Retained samples per stimulus", loc="left")
    ax2[0].grid(True, axis="y", alpha=0.25); ax2[0].set_axisbelow(True)
    panel(ax2[0], "A", dx=-0.055, dy=1.03)

    ax2[1].bar(xs, q.discarded_percent, color="#C62828", edgecolor="black",
               linewidth=0.4)
    ax2[1].set_ylabel("discarded (%)")
    ax2[1].set_title("B. Discarded by quality filtering", loc="left")
    ax2[1].set_xticks(xs)
    ax2[1].set_xticklabels(q.label, rotation=90, fontsize=7)
    ax2[1].set_xlabel("cognitive task")
    ax2[1].grid(True, axis="y", alpha=0.25); ax2[1].set_axisbelow(True)
    panel(ax2[1], "B", dx=-0.055, dy=1.03)

    f2.suptitle(f"Supplementary Figure: per-stimulus data retention after "
                f"quality filtering ({len(subjects)} participants)",
                fontsize=12, fontweight="bold", y=0.94)
    out2 = os.path.join(RES, "SuppFig_stimulus_quality.pdf")
    f2.savefig(out2, bbox_inches="tight")
    f2.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out2} (+ .png)")
    print(f"  {len(q)} stimuli, kept {q.kept.sum():,} of {q.total.sum():,} rows "
          f"({100 * q.kept.sum() / q.total.sum():.1f}%)")


if __name__ == "__main__":
    main()
