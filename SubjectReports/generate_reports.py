"""
generate_reports.py
===================
Generate a professional per-subject Word document brain-age report.

Each report includes:
  - Subject demographics (ID, true age, predicted brain age, brain age gap)
  - Overall prediction summary with context
  - Lobe-specific predicted ages (based on stimulus→primary lobe mapping)
  - Cognitive domain summary table
  - Per-stimulus predicted age bar chart (best/worst tasks)
  - Lobe comparison bar chart
  - FFT band power heatmap per lobe (from per-subject FFT CSV)
  - Lobe-level attention heatmap for the subject's age group (Head 1 & Head 2)
  - Attention-weighted FFT connectivity (per-subject FFT × age-group attention)

Outputs:
  SubjectReports/<ID>_brain_age_report.docx
"""

import os, sys, json, glob, io, warnings, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAINAGE   = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

RESULTS_CSV  = os.path.join(_BRAINAGE, "Modeling_ContrastiveChannelWise", "results.csv")
STIM_CSV     = os.path.join(_BRAINAGE, "stimuli_descriptions_and_expectations.csv")
META_XLS     = os.path.join(_BRAINAGE, "data", "PilotDatasetSubjects.xlsx")
RESULTS_DIR  = os.path.join(_BRAINAGE, "results")
ATT_CACHE    = os.path.join(_BRAINAGE, "MainFigures", "F6", "attention_cache_4bin.pkl")
OUT_DIR      = _SCRIPT_DIR

# Age group boundaries (must match attention cache generation)
AGE_GROUPS = [(0, 25), (25, 31), (31, 39), (39, 47), (47, 200)]
AGE_BIN_LABELS = ["<25", "25–30", "31–38", "39–46", "47+"]

def age_group_idx(age):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(AGE_GROUPS) - 1

# ── Palette & constants ───────────────────────────────────────────────────────
LOBE_PAL = {
    "Frontal":   "#4C72B0",
    "Parietal":  "#55A868",
    "Temporal":  "#DD8452",
    "Occipital": "#C44E52",
    "Frontal-Central": "#8172B2",
}
DOMAIN_PAL = {
    "Speech production":               "#E76F51",
    "Semantic verbal fluency":         "#F4A261",
    "Pattern recognition reasoning":   "#E9C46A",
    "Memorization":                    "#2A9D8F",
    "Recall":                          "#264653",
    "Executive reasoning":             "#4C72B0",
    "Spatial attention":               "#55A868",
    "Spatial imagery and navigation":  "#8172B2",
    "Motor coordination":              "#C44E52",
    "Auditory perception":             "#DD8452",
    "Resting state attention":         "#B0B0B0",
    "Resting state baseline":          "#D0D0D0",
    "Language comprehension":          "#6FA8DC",
}
DEFAULT_COLOUR = "#A0A0A0"

BANDS     = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]
LOBES     = ["Frontal", "Temporal", "Parietal", "Occipital"]
REGION_MAP = {
    "AF3":"Frontal","F7":"Frontal","F3":"Frontal","FC5":"Frontal",
    "FC6":"Frontal","F4":"Frontal","F8":"Frontal","AF4":"Frontal",
    "T7":"Temporal","T8":"Temporal",
    "P7":"Parietal","P8":"Parietal",
    "O1":"Occipital","O2":"Occipital",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8F9FA",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Load shared data ──────────────────────────────────────────────────────────
print("Loading prediction results …")
results = pd.read_csv(RESULTS_CSV)

print("Loading stimuli descriptions …")
stim_df  = pd.read_csv(STIM_CSV)
stim_map = stim_df.set_index("original_stimuli_name")

# Allowed task stimuli: only those in the CSV, minus baselines and instruction stimuli
_EXCLUDE = {"eyesopen", "eyesclose", "instruction_listening"}
TASK_STIMULI = set(stim_df["original_stimuli_name"]) - _EXCLUDE

print("Loading subject metadata …")
meta = pd.read_excel(META_XLS, header=1)
meta["ID"] = meta["ID"].astype(str).str.strip()

print("Loading attention cache …")
with open(ATT_CACHE, "rb") as _f:
    _att_cache = pickle.load(_f)

# Build channel label list and lobe→channel index map (matches cache generation)
import json as _json
with open(os.path.join(_BRAINAGE, "config.json")) as _cf:
    _CH_FULL = _json.load(_cf)["EEG_columns"]
CH_LABELS = [c.replace("EEG.", "") for c in _CH_FULL]   # 14 labels
lobe_idx  = {lobe: [i for i, ch in enumerate(CH_LABELS) if REGION_MAP.get(ch) == lobe]
             for lobe in LOBES}

# Keep only valid task stimuli (in CSV, no baselines, no instructions)
results = results[results["stimulus"].isin(TASK_STIMULI)].copy()

# Merge predictions with stimulus info
results = results.merge(
    stim_df[["original_stimuli_name", "cognitive_domain", "primary_lobe_targeted", "description"]],
    left_on="stimulus", right_on="original_stimuli_name", how="left"
)

# ── Task display names (human-readable, from description field) ───────────────
def _make_display_name(stim_name, desc):
    """Human-readable label from the stimulus description (up to ~10 words, title-cased)."""
    if pd.isna(desc) or not str(desc).strip():
        return stim_name.replace("_", " ").title()
    words = str(desc).split()
    # Strip leading "Participants " if present
    if words and words[0].lower() == "participants":
        words = words[1:]
    # Capitalise first word, keep rest as-is
    if words:
        words[0] = words[0].capitalize()
    label = " ".join(words[:10])
    if len(words) > 10:
        label += "…"
    return label

_stim_display = {
    row["original_stimuli_name"]: _make_display_name(
        row["original_stimuli_name"], row.get("description", "")
    )
    for _, row in stim_df.iterrows()
}

def task_label(stim_name):
    return _stim_display.get(stim_name, stim_name.replace("_", " ").title())

# ── Pre-compute dataset-level Head 1 attention per lobe (PASA baseline) ───────
# Used for the PASA comparison chart; keyed by age_group_idx
_h1_sent_by_age = {}   # ag_idx -> np.array of shape (n_lobes,)
for _ag_idx in sorted({k[1] for k in _att_cache}):
    _mats = [v for (s, a, hh), v in _att_cache.items() if a == _ag_idx and hh == 0]
    if not _mats:
        continue
    _m = np.mean(_mats, axis=0)
    _sent = np.zeros(len(LOBES))
    for si, src in enumerate(LOBES):
        for ti, tgt in enumerate(LOBES):
            if si == ti:
                continue
            pairs = [(i, j) for i in lobe_idx[src] for j in lobe_idx[tgt]]
            if pairs:
                _sent[si] += np.mean([_m[i, j] for i, j in pairs])
    _h1_sent_by_age[_ag_idx] = _sent

_all_ag_ids = sorted(_h1_sent_by_age.keys())
_h1_sent_dataset_mean = np.mean([_h1_sent_by_age[a] for a in _all_ag_ids], axis=0)

# ── Pre-compute per-subject summary stats (for cohort positioning) ────────────
_cohort = (results.groupby("ID")
           .agg(true_age=("true_age", "first"),
                predicted_age=("predicted_age", "mean"))
           .reset_index())
_cohort["gap"]       = _cohort["predicted_age"] - _cohort["true_age"]
_cohort["age_group"] = _cohort["true_age"].apply(age_group_idx)

# ── Helpers ───────────────────────────────────────────────────────────────────
def buf_to_stream(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color.lstrip("#"))
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def add_heading(doc, text, level=1, colour=RGBColor(0x1A, 0x53, 0x76)):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.color.rgb = colour
    return h


def add_kv_table(doc, rows, col_widths=(4.5, 8.5)):
    tbl = doc.add_table(rows=len(rows), cols=2)
    tbl.style = "Table Grid"
    for i, (key, val) in enumerate(rows):
        tbl.rows[i].cells[0].text = key
        tbl.rows[i].cells[1].text = str(val)
        tbl.rows[i].cells[0].paragraphs[0].runs[0].bold = True
        tbl.rows[i].cells[0].width = Cm(col_widths[0])
        tbl.rows[i].cells[1].width = Cm(col_widths[1])
    return tbl


# ── Figure 0: Cohort positioning ─────────────────────────────────────────────
def fig_cohort_position(subj_id, true_age, pred_age, ag_idx):
    """
    Shows where this participant sits relative to:
      • The full cohort (all 77 subjects) — grey dots
      • Their own age group — coloured dots
      • Themselves — highlighted star

    Left panel  : Brain Age Gap (predicted − true age) — positive = older brain
    Right panel : Predicted brain age vs true age scatter for the cohort,
                  with a reference diagonal (perfect prediction) and this subject marked.
    """
    subj_gap  = pred_age - true_age
    bin_data  = _cohort[_cohort["age_group"] == ag_idx]
    all_data  = _cohort

    cohort_mean_gap  = all_data["gap"].mean()
    bin_mean_gap     = bin_data["gap"].mean()
    cohort_mean_pred = all_data["predicted_age"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: Brain Age Gap strip + distribution ──────────────────────────────
    ax = axes[0]
    rng = np.random.default_rng(42)

    # All cohort — jittered grey dots
    jitter_all = rng.uniform(-0.12, 0.12, len(all_data))
    ax.scatter(all_data["gap"], jitter_all, color="#CCCCCC", s=28,
               zorder=2, label="All participants", alpha=0.7)

    # Same age bin — coloured dots
    jitter_bin = rng.uniform(-0.12, 0.12, len(bin_data))
    bin_colour = "#4C72B0"
    ax.scatter(bin_data["gap"], jitter_bin, color=bin_colour, s=40,
               zorder=3, label=f"Same age group ({AGE_BIN_LABELS[ag_idx]} yr)", alpha=0.85)

    # This subject — large star
    ax.scatter([subj_gap], [0], color="#E63946", s=260, marker="*",
               zorder=5, label=f"This participant ({subj_id})", edgecolors="white", linewidth=0.8)

    # Reference lines
    ax.axvline(0,               color="#333", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)
    ax.axvline(cohort_mean_gap, color="#888", linewidth=1.5, linestyle=":",  alpha=0.8, zorder=1)
    ax.axvline(bin_mean_gap,    color=bin_colour, linewidth=1.5, linestyle=":", alpha=0.8, zorder=1)

    ax.text(cohort_mean_gap, 0.18, f" cohort\n avg {cohort_mean_gap:+.1f}",
            fontsize=7, color="#666", va="bottom")
    ax.text(bin_mean_gap,   -0.22, f" age group\n avg {bin_mean_gap:+.1f}",
            fontsize=7, color=bin_colour, va="top")
    ax.text(subj_gap + 0.3, 0, f"  {subj_gap:+.1f} yr", fontsize=9, fontweight="bold",
            color="#E63946", va="center")

    ax.set_xlabel("Brain Age Gap  (predicted − true age, years)", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Where does this participant's brain age gap\nfit in the cohort?",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    ax.set_ylim(-0.5, 0.5)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    # ── Right: Predicted vs true age scatter ──────────────────────────────────
    ax2 = axes[1]
    age_range = [_cohort["true_age"].min() - 2, _cohort["true_age"].max() + 2]

    # Perfect-prediction diagonal
    ax2.plot(age_range, age_range, color="#AAAAAA", linewidth=1.2,
             linestyle="--", label="Perfect prediction", zorder=1)

    # All cohort
    other = _cohort[_cohort["ID"] != subj_id]
    ax2.scatter(other["true_age"], other["predicted_age"],
                c=[LOBE_PAL.get("Frontal") if r == ag_idx else "#CCCCCC"
                   for r in other["age_group"]],
                s=30, alpha=0.7, zorder=2)

    # Same age bin highlight (re-draw on top)
    ax2.scatter(bin_data[bin_data["ID"] != subj_id]["true_age"],
                bin_data[bin_data["ID"] != subj_id]["predicted_age"],
                color=bin_colour, s=38, alpha=0.85, zorder=3,
                label=f"Same age group ({AGE_BIN_LABELS[ag_idx]} yr)")

    # This subject
    ax2.scatter([true_age], [pred_age], color="#E63946", s=280, marker="*",
                zorder=5, edgecolors="white", linewidth=0.8,
                label=f"This participant ({subj_id})")

    # Cohort mean predicted age line
    ax2.axhline(cohort_mean_pred, color="#999", linewidth=1.0, linestyle=":",
                alpha=0.7, label=f"Cohort mean prediction ({cohort_mean_pred:.1f} yr)")

    ax2.set_xlabel("True age (years)", fontsize=9)
    ax2.set_ylabel("Predicted brain age (years)", fontsize=9)
    ax2.set_title("Predicted brain age vs true age\n(all participants in cohort)",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7.5, frameon=True, loc="upper left")
    ax2.set_xlim(age_range)
    ax2.set_ylim(age_range)
    ax2.grid(linewidth=0.3, alpha=0.4)
    ax2.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    fig.suptitle(f"Participant {subj_id} — position within the study cohort",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 1: Per-stimulus predicted age (horizontal bar, top/bottom N) ───────
def fig_stimulus_bars(subj_stim, true_age, top_n=8):
    # Remove stimuli with very few windows (unreliable)
    subj_stim = subj_stim[subj_stim["count"] >= 50].copy()
    subj_stim = subj_stim.sort_values("mean")

    best  = subj_stim.head(top_n)
    worst = subj_stim.tail(top_n)
    plot_df = pd.concat([best, worst]).drop_duplicates()
    plot_df = plot_df.sort_values("mean")

    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.42)))

    colours = []
    for _, row in plot_df.iterrows():
        dom = row.get("cognitive_domain", "")
        colours.append(DOMAIN_PAL.get(dom, DEFAULT_COLOUR))

    bars = ax.barh(range(len(plot_df)), plot_df["mean"].values,
                   xerr=plot_df["std"].values, color=colours,
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(ecolor="#555", elinewidth=0.8, capsize=2),
                   height=0.7)

    ax.axvline(true_age, color="#CC2222", linewidth=1.8,
               linestyle="--", label=f"True age ({true_age:.0f} yr)", zorder=5)
    ax.axvline(subj_stim["mean"].mean(), color="#2244AA", linewidth=1.4,
               linestyle=":", label=f"Overall predicted ({subj_stim['mean'].mean():.1f} yr)", zorder=5)

    labels = []
    for stim_name, row in plot_df.iterrows():
        nm = task_label(stim_name)
        if len(nm) > 38:
            nm = nm[:36] + "…"
        labels.append(nm)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted brain age (years)", fontsize=9)
    ax.set_title("Per-task predicted brain age\n(best ↔ worst performing tasks)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Colour legend for domains
    seen_domains = plot_df["cognitive_domain"].dropna().unique()
    patches = [mpatches.Patch(color=DOMAIN_PAL.get(d, DEFAULT_COLOUR), label=d)
               for d in seen_domains]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.18),
               title="Cognitive domain")

    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 2: Lobe predicted age bar chart ───────────────────────────────────
def fig_lobe_bars(lobe_ages, true_age, overall_pred):
    lobes = list(lobe_ages.keys())
    vals  = list(lobe_ages.values())
    cols  = [LOBE_PAL.get(l, DEFAULT_COLOUR) for l in lobes]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(lobes, vals, color=cols, edgecolor="white", linewidth=0.6, width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(true_age, color="#CC2222", linewidth=1.8, linestyle="--",
               label=f"True age ({true_age:.0f} yr)")
    ax.axhline(overall_pred, color="#2244AA", linewidth=1.4, linestyle=":",
               label=f"Overall predicted ({overall_pred:.1f} yr)")

    ax.set_ylabel("Mean predicted brain age (years)", fontsize=9)
    ax.set_title("Lobe-specific predicted brain age", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(max(0, min(vals + [true_age, overall_pred]) - 5),
                max(vals + [true_age, overall_pred]) + 5)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 3: FFT spectral profile — radar + grouped bars ────────────────────
def fig_fft_profile(fft_df):
    """
    Left  — Radar/spider chart: one shape per lobe, 5 axes = frequency bands.
             Each value is log10(median power), normalised to [0,1] per band so
             all bands are comparable. Immediately shows the spectral 'fingerprint'
             of each brain region.
    Right — Grouped horizontal bar chart: per-band, one bar per lobe.
             Values are % of total power within each band, making it easy to
             read which region dominates each frequency.
    """
    ch_cols = {(ch, band): f"POW.{ch}.{band}"
               for ch in REGION_MAP for band in BANDS
               if f"POW.{ch}.{band}" in fft_df.columns}

    records = []
    for (ch, band), col in ch_cols.items():
        lobe = REGION_MAP[ch]
        val  = fft_df[col].median()
        records.append({"lobe": lobe, "band": band, "power": val})

    pwr = (pd.DataFrame(records)
           .groupby(["lobe", "band"])["power"]
           .mean()
           .unstack()
           .reindex(index=LOBES, columns=BANDS))
    raw = pwr.values.astype(float)

    # Per-band z-score across lobes: for each band axis, center and scale the 4 lobe values.
    # This shows which lobe is RELATIVELY stronger or weaker in each frequency band
    # compared to the other lobes — removing the global 1/f slope that otherwise
    # makes Theta dominate every axis equally.
    radar_vals = np.zeros_like(raw)
    for j in range(raw.shape[1]):          # iterate over bands (columns)
        col = raw[:, j]
        mu, sd = np.nanmean(col), np.nanstd(col)
        if sd > 0:
            radar_vals[:, j] = (col - mu) / sd
        # scale to [0,1] for clean radar display (shift so min=0)
    r_min = radar_vals.min()
    r_max = radar_vals.max()
    if r_max > r_min:
        radar_vals = (radar_vals - r_min) / (r_max - r_min)

    # For bar chart: per-channel mean power per lobe per band
    # (raw already computed as mean across channels in that lobe via groupby mean)
    pct = raw.copy()

    band_labels_short = ["Theta\n4–8 Hz", "Alpha\n8–12 Hz", "Beta-L\n12–16 Hz",
                          "Beta-H\n20–28 Hz", "Gamma\n30–100 Hz"]
    N = len(BANDS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig = plt.figure(figsize=(12, 4.5))

    # ── Radar chart ───────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(121, polar=True)
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(band_labels_short, fontsize=8)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["low", "", "", "high"], fontsize=6, color="grey")
    ax_r.set_ylim(0, 1.15)
    ax_r.set_facecolor("#F8F9FA")

    for li, lobe in enumerate(LOBES):
        vals = list(radar_vals[li]) + [radar_vals[li, 0]]
        ax_r.plot(angles, vals, linewidth=2.2, color=LOBE_PAL.get(lobe, DEFAULT_COLOUR),
                  label=lobe)
        ax_r.fill(angles, vals, alpha=0.08, color=LOBE_PAL.get(lobe, DEFAULT_COLOUR))

    ax_r.set_title("Brain region spectral fingerprint\n(relative strength per band vs other regions)",
                   fontsize=9, fontweight="bold", pad=18)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=True)

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    ax_b = fig.add_subplot(122)
    n_lobes = len(LOBES)
    group_h = 0.65       # total height per band group
    bar_h   = group_h / n_lobes
    y_base  = np.arange(N) * (group_h + 0.3)

    for li, lobe in enumerate(LOBES):
        ys = y_base + li * bar_h
        ax_b.barh(ys, pct[li], height=bar_h * 0.85,
                  color=LOBE_PAL.get(lobe, DEFAULT_COLOUR),
                  label=lobe, edgecolor="white", linewidth=0.4)
        for y, v in zip(ys, pct[li]):
            if not np.isnan(v) and v > 0:
                ax_b.text(v * 1.05, y, f"{v:.2f}", va="center", fontsize=6.5,
                          color="#333333")

    ytick_pos = y_base + group_h / 2 - bar_h / 2
    ax_b.set_yticks(ytick_pos)
    ax_b.set_yticklabels(["Theta\n(4–8 Hz)", "Alpha\n(8–12 Hz)", "Beta-L\n(12–16 Hz)",
                           "Beta-H\n(20–28 Hz)", "Gamma\n(30–100 Hz)"], fontsize=8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Mean power per electrode  (μV²/Hz, log scale)", fontsize=9)
    ax_b.set_title("Average signal strength per electrode\nby region and frequency band",
                   fontsize=9, fontweight="bold")
    ax_b.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle("EEG spectral activity by brain region", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 4: Brain age gap gauge ─────────────────────────────────────────────
def fig_brain_age_gauge(true_age, pred_age):
    gap = pred_age - true_age
    fig, ax = plt.subplots(figsize=(5, 2.8))

    # Simple horizontal gauge
    bar_range = 20
    cmap_vals = np.linspace(-bar_range, bar_range, 200)
    colours   = plt.cm.RdYlGn_r(np.interp(cmap_vals, [-bar_range, bar_range], [0, 1]))

    for i, (v, c) in enumerate(zip(cmap_vals, colours)):
        ax.barh(0, 1, left=v, height=0.6, color=c, linewidth=0)

    gap_clamped = np.clip(gap, -bar_range, bar_range)
    ax.axvline(gap_clamped, color="black", linewidth=3.5, zorder=5)
    ax.axvline(0, color="white", linewidth=1.5, linestyle="--", zorder=4, alpha=0.8)

    ax.text(gap_clamped, 0.5, f" {gap:+.1f} yr", va="bottom", ha="left" if gap >= 0 else "right",
            fontsize=13, fontweight="bold",
            color="#CC2222" if gap > 2 else "#226600" if gap < -2 else "#555555")

    ax.set_xlim(-bar_range - 1, bar_range + 1)
    ax.set_ylim(-0.6, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Brain Age Gap  (predicted − true age, years)", fontsize=9)
    ax.set_title("Brain Age Gap", fontsize=11, fontweight="bold")

    ax.text(-bar_range, -0.45, "Younger brain", fontsize=8, color="#226600", ha="left")
    ax.text( bar_range, -0.45, "Older brain",   fontsize=8, color="#CC2222", ha="right")

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)

    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 5: Attention flow — Head 1 only, coordinator vs hub ───────────────
def fig_attention_flow(ag_idx):
    """
    Head 1 only (the attention pattern that tracks age-related frontal shift).

    COORDINATOR score — how much a region actively reaches out to others.
    Imagine the brain as a team working on a problem. Some members speak up
    and direct the group — they are coordinators. In the brain, a region with
    a high coordinator score is the one actively pulling in information from
    other regions and driving the overall processing. It is not necessarily
    doing the hard work itself — it is orchestrating.

    HUB score — how much a region is consulted by others.
    A high hub score means other brain regions are constantly sending signals
    to this one — they rely on it, they check in with it. It is the specialist
    in the room that everyone turns to.

    Together, these two scores tell us: who is leading the communication,
    and who is being leaned on.
    """
    mats = [v for (s, a, hh), v in _att_cache.items() if a == ag_idx and hh == 0]
    if not mats:
        return None
    mean_mat = np.mean(mats, axis=0)
    sent     = np.zeros(len(LOBES))
    received = np.zeros(len(LOBES))
    for si, src in enumerate(LOBES):
        for ti, tgt in enumerate(LOBES):
            if si == ti:
                continue
            pairs = [(i, j) for i in lobe_idx[src] for j in lobe_idx[tgt]]
            if pairs:
                w = np.mean([mean_mat[i, j] for i, j in pairs])
                sent[si]     += w
                received[ti] += w

    colours = [LOBE_PAL.get(l, DEFAULT_COLOUR) for l in LOBES]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, (vals, title, note) in zip(axes, [
        (sent,
         "COORDINATOR score\n(who initiates brain communication?)",
         "Higher bar = this region actively reaches out to others more.\n"
         "In aging, the Frontal lobe typically rises here — a sign of PASA."),
        (received,
         "HUB score\n(who gets consulted by other regions?)",
         "Higher bar = other regions rely on this one more.\n"
         "Posterior regions (Parietal, Occipital) are often the hub in younger brains."),
    ]):
        bars = ax.bar(LOBES, vals, color=colours, edgecolor="white", linewidth=0.5, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.set_ylabel("Attention weight (relative units)", fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.text(0.5, -0.22, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="#555555",
                style="italic", wrap=True)

    fig.suptitle(
        "Brain communication pattern for this age group\n"
        "(based on the AI model's learned attention — Head 1)",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure: Cognitive domain radar ───────────────────────────────────────────
_DOMAIN_SHORT = {
    "Speech production":               "Speech\nproduction",
    "Semantic verbal fluency":         "Verbal\nfluency",
    "Pattern recognition reasoning":   "Pattern\nrecognition",
    "Memorization":                    "Memory",
    "Recall":                          "Recall",
    "Executive reasoning":             "Executive\nreasoning",
    "Spatial attention":               "Spatial\nattention",
    "Spatial imagery and navigation":  "Spatial\nnavigation",
    "Motor coordination":              "Motor\ncoordination",
    "Auditory perception":             "Auditory\nperception",
    "Resting state attention":         "Resting\nstate",
    "Language comprehension":          "Language",
}

def fig_domain_radar(subj_results, true_age):
    """
    Spider/radar with one axis per cognitive domain.
    The dashed red circle marks the participant's TRUE age.
    The coloured polygon is the participant's PREDICTED brain age per domain.
      - Polygon vertex OUTSIDE the red circle = brain appears OLDER in that domain
      - Polygon vertex INSIDE  the red circle = brain appears YOUNGER in that domain
    Green fill = younger regions; red fill = older regions.
    """
    domain_ages = (subj_results
                   .groupby("cognitive_domain")["predicted_age"]
                   .mean()
                   .dropna())
    if len(domain_ages) < 3:
        return None

    domains = list(domain_ages.index)
    values  = np.array([domain_ages[d] for d in domains], dtype=float)
    labels  = [_DOMAIN_SHORT.get(d, d) for d in domains]

    N = len(domains)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed  = angles + angles[:1]
    values_closed  = list(values) + [values[0]]
    true_closed    = [true_age] * N + [true_age]

    # Axis limits: give a bit of room above/below
    ax_min = max(0, min(values.min(), true_age) - 6)
    ax_max = max(values.max(), true_age) + 6

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Reference rings at true age ± 5 yr
    for delta, lw, ls, alpha in [(0, 2.5, "--", 1.0), (-5, 0.8, ":", 0.5), (5, 0.8, ":", 0.5)]:
        ring = [true_age + delta] * N + [true_age + delta]
        ax.plot(angles_closed, ring, color="#E63946", linewidth=lw, linestyle=ls, alpha=alpha)

    # Fill younger (inside) / older (outside) regions
    vals_arr = np.array(values_closed)
    true_arr = np.array(true_closed)

    # Younger region (polygon clipped to true_age circle) — green
    younger = np.minimum(vals_arr, true_arr)
    ax.fill(angles_closed, younger, color="#55A868", alpha=0.18, zorder=1)

    # Older region (polygon above true_age circle) — red
    older = np.maximum(vals_arr, true_arr)
    ax.fill_between(angles_closed, true_arr, older,
                    where=(older > true_arr),
                    color="#C44E52", alpha=0.22, zorder=2)

    # Main polygon
    ax.plot(angles_closed, values_closed, color="#2244AA",
            linewidth=2.2, zorder=3)
    ax.fill(angles_closed, values_closed, color="#2244AA", alpha=0.08, zorder=2)

    # Mark vertices
    for ang, val, dom in zip(angles, values, domains):
        gap = val - true_age
        colour = "#C44E52" if gap > 0.5 else "#55A868" if gap < -0.5 else "#888888"
        ax.scatter([ang], [val], s=60, color=colour, zorder=5, edgecolors="white", linewidth=0.8)
        # Annotate gap
        r_text = val + (ax_max - ax_min) * 0.07
        ax.annotate(f"{gap:+.1f} yr",
                    xy=(ang, val), xytext=(ang, r_text),
                    fontsize=7, color=colour, fontweight="bold",
                    ha="center", va="center")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(ax_min, ax_max)

    # Y-axis ticks: show true age prominently
    yticks = [round(true_age - 5), round(true_age), round(true_age + 5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t} yr" for t in yticks], fontsize=7, color="#666")

    ax.grid(color="#CCCCCC", linewidth=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#E63946", lw=2.5, linestyle="--", label=f"True age ({true_age:.0f} yr)"),
        mpatches.Patch(color="#55A868", alpha=0.5, label="Younger than true age"),
        mpatches.Patch(color="#C44E52", alpha=0.5, label="Older than true age"),
        Line2D([0], [0], color="#2244AA", lw=2.2, label="Predicted brain age"),
    ]
    ax.legend(handles=legend_els, loc="upper right",
              bbox_to_anchor=(1.35, 1.1), fontsize=8.5, frameon=True)

    ax.set_title("Predicted brain age by cognitive domain\n"
                 "(vs true age — dashed red circle)",
                 fontsize=10, fontweight="bold", pad=20)
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 5b: Per-subject PASA indicator from model predictions ───────────────
def fig_pasa_per_subject(subj_id, subj_results, true_age):
    """
    Computes a per-subject PASA indicator directly from the model's predictions:
      - Frontal predicted age  = mean predicted age on Frontal-targeted tasks
      - Posterior predicted age = mean predicted age on Parietal + Occipital tasks
      - PASA index = Frontal − Posterior  (positive = Frontal appears older)

    Left panel  — per-lobe predicted ages for THIS participant vs cohort mean,
                  with true age as reference.
    Right panel — PASA index for this participant placed on the cohort distribution
                  (strip plot); shows whether their Frontal-Posterior gap is
                  typical, high, or low for the cohort.
    """
    POSTERIOR = ["Parietal", "Occipital"]

    # ── Per-lobe ages for this participant ────────────────────────────────────
    subj_lobe = (subj_results
                 .groupby("primary_lobe_targeted")["predicted_age"]
                 .mean())

    frontal_pred  = subj_lobe.get("Frontal",  np.nan)
    post_pred     = np.nanmean([subj_lobe.get(l, np.nan) for l in POSTERIOR])
    subj_pasa_idx = frontal_pred - post_pred

    # ── Same for every cohort participant ─────────────────────────────────────
    cohort_lobe = results.groupby(["ID", "primary_lobe_targeted"])["predicted_age"].mean().unstack()
    cohort_lobe["posterior"] = cohort_lobe[POSTERIOR].mean(axis=1)
    cohort_lobe["pasa_idx"]  = cohort_lobe.get("Frontal", np.nan) - cohort_lobe["posterior"]
    cohort_mean_lobe = cohort_lobe.mean()

    all_lobes_plot = [l for l in LOBES if l in subj_lobe.index or l in cohort_mean_lobe.index]

    colours = [LOBE_PAL.get(l, DEFAULT_COLOUR) for l in all_lobes_plot]
    x = np.arange(len(all_lobes_plot))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: per-lobe predicted age — this participant vs cohort mean ────────
    ax = axes[0]
    cohort_vals = [cohort_mean_lobe.get(l, np.nan) for l in all_lobes_plot]
    subj_vals   = [subj_lobe.get(l, np.nan) for l in all_lobes_plot]

    b1 = ax.bar(x - w/2, cohort_vals, width=w, color="#CCCCCC",
                edgecolor="white", linewidth=0.5, label="Cohort average")
    b2 = ax.bar(x + w/2, subj_vals,   width=w, color=colours,
                edgecolor="white", linewidth=0.5, label="This participant")

    ax.axhline(true_age, color="#E63946", linewidth=1.8, linestyle="--",
               label=f"True age ({true_age:.0f} yr)")

    for bar, v in zip(b2, subj_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_lobes_plot, fontsize=9)
    ax.set_ylabel("Mean predicted brain age (years)", fontsize=8.5)
    ax.set_title("This participant's predicted brain age\nper region vs cohort average",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Right: PASA index distribution across all participants ────────────────
    ax2 = axes[1]
    pasa_vals = cohort_lobe["pasa_idx"].dropna().values
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.18, 0.18, len(pasa_vals))
    cohort_mean_pasa = np.nanmean(pasa_vals)

    ax2.scatter(pasa_vals, jitter, color="#CCCCCC", s=28, alpha=0.7,
                zorder=2, label="Other participants")
    ax2.scatter([subj_pasa_idx], [0], color="#E63946", s=260, marker="*",
                zorder=5, edgecolors="white", linewidth=0.8,
                label=f"This participant ({subj_pasa_idx:+.1f} yr)")
    ax2.axvline(0, color="#444", linewidth=1.0, linestyle="--", alpha=0.5)
    ax2.axvline(cohort_mean_pasa, color="#888", linewidth=1.5, linestyle=":",
                label=f"Cohort mean ({cohort_mean_pasa:+.1f} yr)")

    ax2.text(subj_pasa_idx, 0.22, f" {subj_pasa_idx:+.1f} yr",
             fontsize=9, fontweight="bold", color="#E63946", va="bottom")

    ax2.set_xlabel("PASA index  (Frontal predicted age − Posterior predicted age, years)",
                   fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_title("PASA index: how much older does the\nFrontal lobe appear vs posterior regions?",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, frameon=True, loc="upper left")
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)

    fig.suptitle(
        "Frontal vs Posterior brain age — a personalised PASA indicator\n"
        "(Posterior-to-Anterior Shift in Aging: Frontal takes over with age)",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Figure 6: Attention-weighted FFT connectivity — grouped bars per band ──────
def fig_att_weighted_fft(ag_idx, fft_df):
    """
    One subplot per frequency band (5 total).
    For each band: grouped bar chart where x = source lobe, bars = target lobe.
    Bar height = attention-weighted FFT power (z-scored across all lobe pairs
    within this subject so bars from different bands are comparable in shape).

    Reading it: "When the Frontal lobe pays attention, which regions show the
    strongest signal at each frequency?" Taller bar = stronger directed connectivity.
    """
    mats = [v for (s, a, hh), v in _att_cache.items() if a == ag_idx]
    if not mats:
        return None
    mean_mat = np.mean(mats, axis=0)

    pow_cols = {}
    for ch in CH_LABELS:
        for band in BANDS:
            col = f"POW.{ch}.{band}"
            if col in fft_df.columns:
                pow_cols[(ch, band)] = col

    ch_fft = {ch: {band: fft_df[pow_cols[(ch, band)]].median()
                   if (ch, band) in pow_cols else np.nan
                   for band in BANDS}
              for ch in CH_LABELS}

    # conn[src][tgt][band]
    conn = {src: {tgt: {band: np.nan for band in BANDS} for tgt in LOBES} for src in LOBES}
    for src in LOBES:
        for tgt in LOBES:
            for band in BANDS:
                wsum, wtot = 0.0, 0.0
                for i in lobe_idx[src]:
                    for j in lobe_idx[tgt]:
                        if i == j:
                            continue
                        w = mean_mat[i, j]
                        v = ch_fft[CH_LABELS[j]][band]
                        if not np.isnan(v):
                            wsum += w * v
                            wtot += w
                conn[src][tgt][band] = wsum / wtot if wtot > 0 else np.nan

    # Global z-score per band across all (src,tgt) pairs so bars within a band are comparable
    for band in BANDS:
        vals = [conn[s][t][band] for s in LOBES for t in LOBES if not np.isnan(conn[s][t][band])]
        if len(vals) > 1:
            mu, sd = np.mean(vals), np.std(vals)
            if sd > 0:
                for src in LOBES:
                    for tgt in LOBES:
                        v = conn[src][tgt][band]
                        conn[src][tgt][band] = (v - mu) / sd if not np.isnan(v) else np.nan

    band_full = {"Theta": "Theta  (4–8 Hz)", "Alpha": "Alpha  (8–12 Hz)",
                 "BetaL": "Beta-Low  (12–16 Hz)", "BetaH": "Beta-High  (20–28 Hz)",
                 "Gamma": "Gamma  (30–100 Hz)"}

    n_src    = len(LOBES)
    n_tgt    = len(LOBES)
    bar_w    = 0.18
    x        = np.arange(n_src)

    fig, axes = plt.subplots(1, len(BANDS), figsize=(14, 4.0), sharey=True)

    for ax, band in zip(axes, BANDS):
        for ti, tgt in enumerate(LOBES):
            offsets = (ti - n_tgt / 2 + 0.5) * bar_w
            vals = [conn[src][tgt][band] for src in LOBES]
            ax.bar(x + offsets, vals, width=bar_w * 0.9,
                   color=LOBE_PAL.get(tgt, DEFAULT_COLOUR),
                   label=tgt if band == BANDS[0] else None,
                   edgecolor="white", linewidth=0.3)

        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l[:3]}." for l in LOBES], fontsize=8)
        ax.set_xlabel("Source region", fontsize=8)
        ax.set_title(band_full[band], fontsize=8.5, fontweight="bold")
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if band == BANDS[0]:
            ax.set_ylabel("Connectivity strength (z-score)", fontsize=8)

    # Shared legend — target lobes
    handles = [mpatches.Patch(color=LOBE_PAL.get(l, DEFAULT_COLOUR), label=f"→ {l}")
               for l in LOBES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.08),
               title="Target region (bar colour)")

    fig.suptitle("How strongly does each brain region drive activity elsewhere?\n"
                 "(attention-weighted signal strength, z-scored per frequency band)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return buf_to_stream(fig)


# ── Build one Word document ───────────────────────────────────────────────────
def build_report(subj_id, subj_results, subj_meta, fft_path):
    doc = Document()

    # Page margins
    section = doc.sections[0]
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

    # ── Cover heading ─────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("EEG Brain Age Report")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1A, 0x53, 0x76)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(f"Subject {subj_id}  ·  Confidential Research Record").font.size = Pt(11)

    doc.add_paragraph()  # spacer

    # ── 1. Demographics ───────────────────────────────────────────────────────
    add_heading(doc, "1.  Subject Overview", level=2)

    true_age    = subj_results["true_age"].iloc[0]
    overall_pred = subj_results["predicted_age"].mean()
    gap          = overall_pred - true_age
    n_tasks      = subj_results["stimulus"].nunique()
    n_windows    = len(subj_results)

    gender = age_label = edu = "—"
    if subj_meta is not None:
        gender    = str(subj_meta.get("Gender", "—"))
        edu       = str(subj_meta.get("Education", "—"))

    gap_str = f"{gap:+.1f} yr  ({'older' if gap > 0 else 'younger'} brain)"
    add_kv_table(doc, [
        ("Subject ID",              subj_id),
        ("Chronological age",       f"{true_age:.0f} years"),
        ("Predicted brain age",     f"{overall_pred:.1f} years"),
        ("Brain age gap",           gap_str),
        ("Gender",                  gender),
        ("Education",               edu),
        ("Tasks performed",         f"{n_tasks}"),
        ("EEG windows analysed",    f"{n_windows:,}"),
    ])
    doc.add_paragraph()

    # Brain age gap gauge
    doc.add_paragraph("Brain Age Gap Visualisation:").runs[0].bold = True
    gauge_stream = fig_brain_age_gauge(true_age, overall_pred)
    doc.add_picture(gauge_stream, width=Inches(5.0))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Interpretive blurb
    p = doc.add_paragraph()
    if abs(gap) <= 2:
        interp = (f"The model predicts a brain age of {overall_pred:.1f} years, which is within "
                  f"±2 years of the chronological age ({true_age:.0f} yr). "
                  "This is consistent with typical brain aging trajectories.")
    elif gap > 2:
        interp = (f"The model predicts a brain age of {overall_pred:.1f} years, "
                  f"{gap:.1f} years older than the chronological age ({true_age:.0f} yr). "
                  "A positive brain age gap may reflect accelerated neural aging patterns "
                  "under the tasks assessed.")
    else:
        interp = (f"The model predicts a brain age of {overall_pred:.1f} years, "
                  f"{abs(gap):.1f} years younger than the chronological age ({true_age:.0f} yr). "
                  "A negative brain age gap suggests the EEG neural signature appears "
                  "younger than chronological age — often associated with efficient brain function.")
    p.add_run(interp).font.size = Pt(10)
    doc.add_paragraph()

    # ── 2. Lobe-specific brain age ────────────────────────────────────────────
    add_heading(doc, "2.  Lobe-Specific Predicted Brain Age", level=2)
    p = doc.add_paragraph(
        "Brain age is estimated separately for tasks whose primary targeted lobe matches each region. "
        "This provides a coarse spatial decomposition of neural age signatures."
    )
    p.runs[0].font.size = Pt(10)

    lobe_ages = (subj_results
                 .groupby("primary_lobe_targeted")["predicted_age"]
                 .mean()
                 .dropna()
                 .to_dict())

    if lobe_ages:
        lobe_stream = fig_lobe_bars(lobe_ages, true_age, overall_pred)
        doc.add_picture(lobe_stream, width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Table
        tbl = doc.add_table(rows=1 + len(lobe_ages), cols=3)
        tbl.style = "Table Grid"
        hdr = tbl.rows[0].cells
        for j, h in enumerate(["Lobe / Region", "Mean predicted age (yr)", "Δ from true age"]):
            hdr[j].text = h
            hdr[j].paragraphs[0].runs[0].bold = True
            set_cell_bg(hdr[j], "#1A5376")
            hdr[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        for i, (lobe, val) in enumerate(sorted(lobe_ages.items(), key=lambda x: x[1])):
            row = tbl.rows[i + 1].cells
            row[0].text = lobe
            row[1].text = f"{val:.1f}"
            d = val - true_age
            row[2].text = f"{d:+.1f}"
            if i % 2 == 0:
                set_cell_bg(row[0], "EBF5FB")
                set_cell_bg(row[1], "EBF5FB")
                set_cell_bg(row[2], "EBF5FB")
    doc.add_paragraph()

    # ── 3. Cognitive domain summary ───────────────────────────────────────────
    add_heading(doc, "3.  Cognitive Domain Summary", level=2)
    p = doc.add_paragraph(
        "Mean predicted brain age grouped by cognitive domain. "
        "Domains where the model predicts a younger age suggest more efficient neural processing "
        "relative to the subject's true age."
    )
    p.runs[0].font.size = Pt(10)

    domain_ages = (subj_results
                   .groupby("cognitive_domain")["predicted_age"]
                   .mean()
                   .dropna()
                   .sort_values())

    tbl = doc.add_table(rows=1 + len(domain_ages), cols=3)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for j, h in enumerate(["Cognitive domain", "Mean predicted age (yr)", "Δ from true age"]):
        hdr[j].text = h
        hdr[j].paragraphs[0].runs[0].bold = True
        set_cell_bg(hdr[j], "#1A5376")
        hdr[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, (dom, val) in enumerate(domain_ages.items()):
        row = tbl.rows[i + 1].cells
        row[0].text = dom if dom else "—"
        row[1].text = f"{val:.1f}"
        d = val - true_age
        row[2].text = f"{d:+.1f}"
        if i % 2 == 0:
            set_cell_bg(row[0], "EBF5FB")
            set_cell_bg(row[1], "EBF5FB")
            set_cell_bg(row[2], "EBF5FB")
    doc.add_paragraph()

    # ── 4. Best / worst tasks ─────────────────────────────────────────────────
    add_heading(doc, "4.  Best and Worst Performing Tasks", level=2)
    p = doc.add_paragraph(
        "Per-task predictions are averaged across all EEG windows for that stimulus. "
        "Tasks are ranked by predicted brain age: lower predicted age = 'younger' neural signature."
    )
    p.runs[0].font.size = Pt(10)

    subj_stim = (subj_results
                 .groupby(["stimulus", "cognitive_domain"])["predicted_age"]
                 .agg(mean="mean", std="std", count="count")
                 .reset_index()
                 .set_index("stimulus"))
    subj_stim = subj_stim[subj_stim["count"] >= 50].sort_values("mean")

    if not subj_stim.empty:
        stim_stream = fig_stimulus_bars(subj_stim, true_age, top_n=8)
        doc.add_picture(stim_stream, width=Inches(6.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

    if not subj_stim.empty:
        add_heading(doc, "Top 5 Best Tasks (youngest predicted age)", level=3)
        best5 = subj_stim.head(5)
        tbl = doc.add_table(rows=1 + len(best5), cols=4)
        tbl.style = "Table Grid"
        for j, h in enumerate(["Task", "Domain", "Pred. age (yr)", "Δ from true"]):
            tbl.rows[0].cells[j].text = h
            tbl.rows[0].cells[j].paragraphs[0].runs[0].bold = True
            set_cell_bg(tbl.rows[0].cells[j], "#1A7A4A")
            tbl.rows[0].cells[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        for i, (stim_name, row) in enumerate(best5.iterrows()):
            cells = tbl.rows[i + 1].cells
            cells[0].text = task_label(stim_name)
            cells[1].text = row.get("cognitive_domain", "—")
            cells[2].text = f"{row['mean']:.1f}"
            cells[3].text = f"{row['mean'] - true_age:+.1f}"
            if i % 2 == 0:
                for c in cells: set_cell_bg(c, "E8F8F0")
        doc.add_paragraph()

    # ── 4b. Cognitive domain profile (radar) — between best and worst tables ──
    add_heading(doc, "4b.  Cognitive Domain Profile", level=2)
    p = doc.add_paragraph(
        "The spider chart below shows the predicted brain age for each type of cognitive "
        "task this participant performed. The dashed red circle marks the participant's "
        "true age.\n\n"
        "How to read it:\n"
        "  • A point on the web OUTSIDE the red circle means the brain appeared OLDER "
        "than the true age during tasks in that domain — the neural pattern resembled "
        "someone from an older age group.\n"
        "  • A point INSIDE the red circle means the brain appeared YOUNGER — a stronger, "
        "more youthful signal for those tasks.\n"
        "  • The number next to each point (for example +3.2 or −1.8) is the gap in "
        "years between predicted age and true age for that domain.\n\n"
        "A perfectly average brain would form a circle exactly on the red dashed line. "
        "Most people show a mix — performing younger in some areas and older in others. "
        "There is no single 'correct' shape; the profile shows where this person's brain "
        "strengths and relative vulnerabilities may lie."
    )
    p.runs[0].font.size = Pt(10)

    domain_radar_stream = fig_domain_radar(subj_results, true_age)
    if domain_radar_stream:
        doc.add_picture(domain_radar_stream, width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    if not subj_stim.empty:
        add_heading(doc, "Top 5 Worst Tasks (oldest predicted age)", level=3)
        worst5 = subj_stim.tail(5).iloc[::-1]
        tbl2 = doc.add_table(rows=1 + len(worst5), cols=4)
        tbl2.style = "Table Grid"
        for j, h in enumerate(["Task", "Domain", "Pred. age (yr)", "Δ from true"]):
            tbl2.rows[0].cells[j].text = h
            tbl2.rows[0].cells[j].paragraphs[0].runs[0].bold = True
            set_cell_bg(tbl2.rows[0].cells[j], "#7A1A1A")
            tbl2.rows[0].cells[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        for i, (stim_name, row) in enumerate(worst5.iterrows()):
            cells = tbl2.rows[i + 1].cells
            cells[0].text = task_label(stim_name)
            cells[1].text = row.get("cognitive_domain", "—")
            cells[2].text = f"{row['mean']:.1f}"
            cells[3].text = f"{row['mean'] - true_age:+.1f}"
            if i % 2 == 0:
                for c in cells: set_cell_bg(c, "FDF0F0")
    doc.add_paragraph()

    # ── 5. FFT band power ─────────────────────────────────────────────────────
    fft_df = None
    if fft_path and os.path.exists(fft_path):
        add_heading(doc, "5.  EEG Spectral Profile", level=2)
        p = doc.add_paragraph(
            "The brain produces electrical activity at different speeds — these are called "
            "frequency bands. Each band is associated with different mental states:\n\n"
            "  • Theta (4–8 Hz) — memory, navigation, light drowsiness\n"
            "  • Alpha (8–12 Hz) — relaxed wakefulness, idling regions\n"
            "  • Beta-Low (12–16 Hz) — active thinking, motor planning\n"
            "  • Beta-High (20–28 Hz) — concentration, alert cognition\n"
            "  • Gamma (30–100 Hz) — high-level processing, sensory binding\n\n"
            "Left chart (spider/radar) — each coloured shape is one brain region. "
            "The five axes represent the five frequency bands. Each axis shows how that "
            "region compares to the other three regions in that specific band — a vertex "
            "pushed outward means this region is relatively stronger than the others "
            "at that frequency. Because each band is scaled independently, the chart "
            "reveals where each region stands out, not which band is globally strongest.\n\n"
            "Right chart — average signal strength per electrode for each region and band "
            "(log scale, so all bands are visible). This shows absolute power levels: "
            "how strong the raw EEG signal is in each frequency for each brain region."
        )
        p.runs[0].font.size = Pt(10)
        fft_df  = pd.read_csv(fft_path)
        fft_df  = fft_df[fft_df["task_name"].isin(TASK_STIMULI)]
        heat_stream = fig_fft_profile(fft_df)
        doc.add_picture(heat_stream, width=Inches(6.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

    # ── 1b. Cohort positioning ────────────────────────────────────────────────
    add_heading(doc, "1b.  Where Does This Participant Sit in the Study Cohort?", level=2)
    p = doc.add_paragraph(
        "The chart below places this participant (red star) among all other participants "
        "in the study. "
        "The left panel shows the Brain Age Gap — how many years older or younger "
        "the model predicts the brain to be compared to the true age. "
        "A gap of 0 is a perfect match. Positive values mean the brain appears older "
        "than it actually is; negative values mean it appears younger. "
        "The right panel shows how well the model tracked everyone, with the diagonal "
        "line representing a perfect match between true age and predicted brain age."
    )
    p.runs[0].font.size = Pt(10)
    cohort_stream = fig_cohort_position(subj_id, true_age, overall_pred, age_group_idx(true_age))
    doc.add_picture(cohort_stream, width=Inches(7.0))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── 6. Brain communication pattern ───────────────────────────────────────
    ag = age_group_idx(true_age)
    add_heading(doc, "6.  How the Brain Communicates Internally", level=2)
    p = doc.add_paragraph(
        "While processing tasks, different brain regions need to share information — "
        "like colleagues in a team passing notes and consulting each other. "
        "The AI model in this study learns to detect which regions are actively "
        "reaching out to others (Coordinator) and which are being consulted (Hub). "
        "\n\n"
        "A Coordinator is a region that drives information flow. Think of it as the "
        "region that 'looks around' at what others are doing and pulls it all together. "
        "A higher Coordinator score does not mean the region works harder — it means "
        "it plays more of a directing, integrating role. "
        "\n\n"
        "A Hub is a region that other parts of the brain send signals to. "
        "A high Hub score means this region is trusted and relied upon by others "
        "— it is the region everyone else checks in with. "
        "\n\n"
        f"The patterns below are averaged across participants in the same age group "
        f"({AGE_BIN_LABELS[ag]} years). As the brain ages, the Frontal lobe typically "
        "rises as a Coordinator — this is the PASA effect "
        "(Posterior-to-Anterior Shift in Aging)."
    )
    p.runs[0].font.size = Pt(10)

    att_stream = fig_attention_flow(ag)
    if att_stream:
        doc.add_picture(att_stream, width=Inches(6.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── 6b. PASA comparison ───────────────────────────────────────────────────
    add_heading(doc, "6b.  Personal PASA Indicator — Frontal vs Posterior Brain Age", level=2)
    p = doc.add_paragraph(
        "This section is specific to this participant. It uses the model's own predictions "
        "— not age-group averages — to answer: does this person's Frontal lobe appear "
        "older than their posterior brain regions?\n\n"
        "Left chart — predicted brain age for each brain region. The grey bars show the "
        "cohort average; the coloured bars show this participant's values. "
        "The red dashed line is the participant's true age. A coloured bar above the red "
        "line means that region's tasks produced a predicted age older than reality — the "
        "brain may be working harder in that area. A bar below the line means the region's "
        "tasks produced a younger-looking signal.\n\n"
        "Right chart — the PASA index: the difference between the Frontal predicted age "
        "and the average of the posterior (Parietal + Occipital) predicted age. "
        "Each dot is one participant in the study. The red star is this participant.\n\n"
        "A positive PASA index (star to the right of zero) means the Frontal lobe's tasks "
        "produced an older predicted age than posterior tasks — consistent with frontal "
        "dominance increasing with age. A negative value means the opposite: posterior "
        "regions show relatively older signatures, typical of younger brains. "
        "A very large positive number indicates a pronounced frontal-dominant pattern."
    )
    p.runs[0].font.size = Pt(10)

    pasa_stream = fig_pasa_per_subject(subj_id, subj_results, true_age)
    if pasa_stream:
        doc.add_picture(pasa_stream, width=Inches(7.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── 6c. How to interpret overall predicted age vs PASA index ─────────────
    add_heading(doc, "6c.  Understanding Your Results: Overall Brain Age vs PASA Index", level=2)
    p = doc.add_paragraph(
        "Two numbers appear throughout this report — the overall predicted brain age and the "
        "PASA index. It is natural to wonder whether they should agree with each other. "
        "They often do not, and that is completely normal. Here is why."
    )
    p.runs[0].font.size = Pt(10)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Overall predicted brain age — the global picture")
    r.bold = True
    r.font.size = Pt(10)
    p = doc.add_paragraph(
        "The overall predicted brain age is computed by averaging the model's predictions across "
        "all tasks this person performed. It is a single number that summarises how the entire "
        "brain, taken as a whole, compares to the study population. Think of it as a general "
        "fitness score. A value younger than the true age (negative gap) means the EEG patterns "
        "across all tasks looked, on average, like someone from a younger age group. A value older "
        "than the true age (positive gap) means the patterns were more typical of an older brain."
    )
    p.runs[0].font.size = Pt(10)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("PASA index — the internal balance")
    r.bold = True
    r.font.size = Pt(10)
    p = doc.add_paragraph(
        "The PASA index does not measure how old or young the brain is in absolute terms. "
        "It measures the relative balance between the front and the back of the brain. "
        "Specifically, it asks: does the Frontal lobe appear older than the rear regions "
        "(Parietal and Occipital)? A positive PASA index simply means the Frontal lobe's "
        "tasks produced a higher predicted age than the posterior regions' tasks — "
        "not that the person has an old brain overall."
    )
    p.runs[0].font.size = Pt(10)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Why they can point in opposite directions")
    r.bold = True
    r.font.size = Pt(10)
    p = doc.add_paragraph(
        "Consider a marathon runner who is in excellent overall shape. Their resting heart rate, "
        "lung capacity, and endurance are all well above average for their age — a genuine "
        "'younger' body by most measures. But if we look closely, we might notice their right "
        "shoulder carries slightly more muscle tension than the left — a small imbalance caused "
        "by years of running with a specific arm swing. The overall fitness score says younger; "
        "the left-right shoulder comparison reveals an asymmetry. Both observations are true "
        "and they are not contradictory — they simply measure different things."
        "\n\n"
        "The same logic applies here. A person can have:"
    )
    p.runs[0].font.size = Pt(10)

    bullet_items = [
        ("Younger overall brain age + positive PASA index",
         "The brain performs better than average across all tasks, but the Frontal lobe "
         "has already started to increase its coordination role relative to posterior regions. "
         "This is the classic healthy aging pattern — efficient overall, with the Frontal "
         "lobe gradually adapting."),
        ("Older overall brain age + negative PASA index",
         "The brain's EEG patterns look older than average on the whole, yet the posterior "
         "regions are still carrying the coordination load — the shift to Frontal dominance "
         "has not yet taken hold. This could reflect an atypical aging trajectory."),
        ("Younger overall brain age + negative PASA index",
         "Both signals agree: the brain appears young and the posterior regions are still "
         "dominant — typical of younger participants."),
        ("Older overall brain age + positive PASA index",
         "Both signals agree: the brain appears older and the Frontal-posterior shift is "
         "already well established — typical of older participants."),
    ]

    for title, body in bullet_items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(title + " — ")
        r.bold = True
        r.font.size = Pt(10)
        p.add_run(body).font.size = Pt(10)

    doc.add_paragraph()
    p = doc.add_paragraph(
        "In short: the overall predicted brain age tells you where the brain sits on the "
        "global aging scale. The PASA index tells you something about the internal "
        "organisation of that brain — specifically, whether the Frontal lobe has started "
        "to take on more of the coordination role that posterior regions typically carry "
        "in youth. Both numbers together give a richer picture than either alone."
    )
    p.runs[0].font.size = Pt(10)
    doc.add_paragraph()

    # ── 7. Brain regions guide ────────────────────────────────────────────────
    add_heading(doc, "7.  Guide to Brain Regions", level=2)
    p = doc.add_paragraph(
        "Throughout this report, brain activity is described in terms of four major regions "
        "(lobes). Here is a plain-language guide to what each region does and why it matters "
        "for brain aging research."
    )
    p.runs[0].font.size = Pt(10)

    lobe_info = [
        ("Frontal",   "#4C72B0",
         "Located at the front of the head, above the forehead.",
         "Decision-making, planning, problem-solving, working memory, "
         "speech production, and controlling impulses. Often called the "
         "'executive' region — it manages and coordinates complex thought.",
         "In healthy aging, the Frontal lobe gradually takes on MORE coordination "
         "work to compensate for changes elsewhere (PASA effect). A rising Frontal "
         "signal with age is considered a normal, adaptive brain response."),
        ("Temporal",  "#DD8452",
         "Located on the sides of the head, near the ears.",
         "Hearing, language understanding, recognising faces and objects, "
         "and long-term memory storage.",
         "Temporal activity reflects auditory and language processing. "
         "Changes here may reflect differences in how the brain processes "
         "language and sound across age groups."),
        ("Parietal",  "#55A868",
         "Located at the top-back of the head.",
         "Integrating sensory information, spatial awareness, attention, "
         "and coordinating sensory signals with movement.",
         "In younger brains, posterior regions like Parietal tend to carry "
         "more of the coordination load. A gradual handover to Frontal is "
         "the hallmark of normal brain aging."),
        ("Occipital", "#C44E52",
         "Located at the very back of the head.",
         "Visual processing — interpreting what the eyes see.",
         "Occipital signals reflect active visual processing. Tasks involving "
         "reading, visual search, or pattern recognition produce strong "
         "Occipital responses."),
    ]

    lobe_tbl = doc.add_table(rows=len(lobe_info), cols=3)
    lobe_tbl.style = "Table Grid"
    for ri, (lobe, colour, location, functions, aging) in enumerate(lobe_info):
        cells = lobe_tbl.rows[ri].cells
        cells[0].text = lobe
        cells[0].paragraphs[0].runs[0].bold = True
        cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(cells[0], colour.lstrip("#"))
        cells[1].text = f"{location}\n\n{functions}"
        cells[2].text = aging
        for c in [cells[1], cells[2]]:
            c.paragraphs[0].runs[0].font.size = Pt(9)
        if ri % 2 == 0:
            set_cell_bg(cells[1], "F5F5F5")
            set_cell_bg(cells[2], "F5F5F5")
    doc.add_paragraph()

    # ── 8. Task reference legend ──────────────────────────────────────────────
    add_heading(doc, "8.  Task Reference", level=2)
    p = doc.add_paragraph(
        "All tasks performed by this participant, with a plain-language description."
    )
    p.runs[0].font.size = Pt(10)

    performed = sorted(subj_results["stimulus"].dropna().unique())
    tbl_leg = doc.add_table(rows=1 + len(performed), cols=3)
    tbl_leg.style = "Table Grid"
    for j, h in enumerate(["Task (internal name)", "What the participant did", "Cognitive domain"]):
        tbl_leg.rows[0].cells[j].text = h
        tbl_leg.rows[0].cells[j].paragraphs[0].runs[0].bold = True
        set_cell_bg(tbl_leg.rows[0].cells[j], "#1A5376")
        tbl_leg.rows[0].cells[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for i, stim_name in enumerate(performed):
        row_info = stim_map.loc[stim_name] if stim_name in stim_map.index else {}
        desc = str(row_info.get("description", "")) if hasattr(row_info, "get") else ""
        dom  = str(row_info.get("cognitive_domain", "—")) if hasattr(row_info, "get") else "—"
        cells = tbl_leg.rows[i + 1].cells
        cells[0].text = stim_name
        cells[1].text = desc if desc and desc != "nan" else "—"
        cells[2].text = dom if dom and dom != "nan" else "—"
        if i % 2 == 0:
            for c in cells: set_cell_bg(c, "EBF5FB")
    doc.add_paragraph()

    # ── 9. Notes & footer ─────────────────────────────────────────────────────
    add_heading(doc, "Notes", level=2)
    notes = doc.add_paragraph()
    notes.add_run(
        "This report is generated automatically from the EEG Brain Age Estimation Framework. "
        "Predictions are produced by a contrastive-pretrained channel-wise encoder followed by "
        "a multi-head cross-channel attention regressor. Brain age gap values represent deviation "
        "from chronological age and should be interpreted in clinical context. "
        "This document is intended for research purposes only. "
        "Attention matrices reflect age-group averages, not individual-level attention weights."
    ).font.size = Pt(9)
    notes.runs[0].font.color.rgb = RGBColor(0x70, 0x70, 0x70)

    hr = doc.add_paragraph("─" * 80)
    hr.runs[0].font.size = Pt(7)
    hr.runs[0].font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.add_run(
        f"EEG Brain Age Research Framework  ·  Subject {subj_id}  ·  "
        "Confidential – For Research Use Only"
    ).font.size = Pt(8)

    return doc


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    subjects = results["ID"].unique()
    meta_dict = meta.set_index("ID").to_dict(orient="index")
    print(f"\nGenerating reports for {len(subjects)} subjects …\n")

    for subj_id in sorted(subjects):
        print(f"  {subj_id} …", end=" ", flush=True)
        subj_results = results[results["ID"] == subj_id].copy()
        subj_meta    = meta_dict.get(subj_id)
        fft_path     = os.path.join(RESULTS_DIR, subj_id, "2_emotiv_fft_values.csv")

        try:
            doc  = build_report(subj_id, subj_results, subj_meta, fft_path)
            out  = os.path.join(OUT_DIR, f"{subj_id}_brain_age_report.docx")
            doc.save(out)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nAll reports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
