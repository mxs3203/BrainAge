"""
F7_lobe_connectivity.py
=======================
Attention-weighted FFT connectivity: SOURCE lobe → each target lobe.

For each (age_bin, band, source→target pair):
  connectivity = sum_{i in source, j in target} att[i,j] * fft_median[j, band]
                 / sum_{i in source, j in target} att[i,j]

  att[i,j]          = attention weight from channel i (source) to channel j (target)
  fft_median[j,band] = median FFT band power of target channel j in that age bin

This captures both the directed attention (how much source attends to target)
and the neural activity (FFT power) at the target — true directed connectivity.

Panels: one per FFT band (Theta | Alpha | BetaL | BetaH | Gamma)
Lines : Frontal→Frontal (self), →Temporal, →Parietal, →Occipital

Config:
  SOURCE_LOBE — lobe whose outgoing attention is measured
  GRAPH_HEAD  — "HEAD1" | "HEAD2" | "HEAD1+HEAD2"
  COG_DOMAIN  — "all" | substring of cognitive_domain
  Z_SCORE     — True: z-score across age bins; False: raw values

Outputs:
  F7_att_weighted_fft_<source>_<head>_<domain>.pdf / .png
"""
import os, sys, json, pickle, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAINAGE   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_F6_DIR     = os.path.join(_SCRIPT_DIR, "..", "F6")
sys.path.insert(0, _BRAINAGE)

N_BINS     = 4   # 3 | 4 | 5 — selects attention_cache_<N>bin.pkl
ATT_CACHE  = os.path.join(_F6_DIR, f"attention_cache_{N_BINS}bin.pkl")
CONFIG_F   = os.path.join(_BRAINAGE, "config.json")
STIM_CSV   = os.path.join(_BRAINAGE, "stimuli_descriptions_and_expectations.csv")
META_XLS   = os.path.join(_BRAINAGE, "data", "PilotDatasetSubjects.xlsx")
RESULTS    = os.path.join(_BRAINAGE, "results")
OUT_DIR    = _SCRIPT_DIR

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_LOBE = "Frontal"      # lobe whose outgoing attention is measured
GRAPH_HEAD  = "HEAD1"  # "HEAD1" | "HEAD2" | "HEAD1+HEAD2"
COG_DOMAIN  = "all"          # "all" | substring of cognitive_domain
Z_SCORE     = True           # z-score across age bins per (target_lobe, band)
# ─────────────────────────────────────────────────────────────────────────────

LOBES = ["Frontal", "Temporal", "Parietal", "Occipital"]
BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]

REGION_PAL = {
    "Frontal":  "#4C72B0",
    "Temporal": "#DD8452",
    "Parietal": "#55A868",
    "Occipital":"#C44E52",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
})

with open(CONFIG_F) as f:
    CHANNELS = json.load(f)["EEG_columns"]
CH_LABELS = [c.replace("EEG.", "") for c in CHANNELS]   # 14 labels

REGION_MAP = {
    "AF3":"Frontal","F7":"Frontal","F3":"Frontal","FC5":"Frontal",
    "T7":"Temporal","P7":"Parietal","O1":"Occipital","O2":"Occipital",
    "P8":"Parietal","T8":"Temporal","FC6":"Frontal","F4":"Frontal",
    "F8":"Frontal","AF4":"Frontal",
}
lobe_idx = {lobe: [i for i, ch in enumerate(CH_LABELS) if REGION_MAP.get(ch) == lobe]
            for lobe in LOBES}

AGE_GROUPS = [(0,25),(25,31),(31,39),(39,47),(47,200)]
_known_lbls = {
    3: ["Young\n(<30)",  "Middle\n(30-50)", "Older\n(>50)"],
    4: ["Q1\n(<27)",     "Q2\n(27-33)",     "Q3\n(34-44)",  "Q4\n(>=45)"],
    5: ["Q1\n(<25)",     "Q2\n(25-30)",     "Q3\n(31-38)",  "Q4\n(39-46)", "Q5\n(>=47)"],
}

def age_group(age):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(AGE_GROUPS) - 1

# ── Stimulus filter ───────────────────────────────────────────────────────────
_stim_df    = pd.read_csv(STIM_CSV)
_BASELINES  = {"eyesopen", "eyesclose"}
_ALLOWED    = set(_stim_df["original_stimuli_name"]) - _BASELINES
_domain_map = dict(zip(_stim_df["original_stimuli_name"],
                       _stim_df["cognitive_domain"].str.lower()))

def stim_ok(s):
    if s not in _ALLOWED:
        return False
    return True if COG_DOMAIN == "all" else COG_DOMAIN.lower() in _domain_map.get(s, "")

# ── Load attention cache ──────────────────────────────────────────────────────
print("Loading attention cache ...")
with open(ATT_CACHE, "rb") as f:
    att = pickle.load(f)

age_grp_ids = sorted({k[1] for k in att})
n_bins      = len(age_grp_ids)
bin_lbls    = _known_lbls.get(n_bins, [f"Bin {ag}" for ag in age_grp_ids])

_valid_heads = {"HEAD1", "HEAD2", "HEAD1+HEAD2"}
if GRAPH_HEAD not in _valid_heads:
    raise ValueError(f"GRAPH_HEAD must be one of {_valid_heads}")

# Mean attention matrix per age bin [n_bins, 14, 14]
att_matrices = {}
for ag_i, ag in enumerate(age_grp_ids):
    if GRAPH_HEAD == "HEAD1+HEAD2":
        mats = [v for (s, a, hh), v in att.items() if a == ag and stim_ok(s)]
    else:
        hh_idx = 0 if GRAPH_HEAD == "HEAD1" else 1
        mats = [v for (s, a, hh), v in att.items() if a == ag and hh == hh_idx and stim_ok(s)]
    att_matrices[ag_i] = np.mean(mats, axis=0) if mats else np.zeros((14, 14))

src_idx = lobe_idx[SOURCE_LOBE]

# ── Load FFT data ─────────────────────────────────────────────────────────────
print("Loading FFT data from results/ ...")
meta = pd.read_excel(META_XLS, header=1)[["ID", "Age"]].dropna()
age_map = dict(zip(meta["ID"].astype(str), meta["Age"].astype(float)))

fft_files = glob.glob(os.path.join(RESULTS, "BA*", "2_emotiv_fft_values.csv"))
fft_parts = []
for fp in fft_files:
    df = pd.read_csv(fp)
    subj_id = df["ID"].iloc[0] if "ID" in df.columns else os.path.basename(os.path.dirname(fp))
    age = age_map.get(str(subj_id))
    if age is None:
        continue
    df["age_group"] = age_group(age)
    df["age"]       = age
    fft_parts.append(df)

fft_all = pd.concat(fft_parts, ignore_index=True)
fft_all = fft_all[fft_all["task_name"].apply(stim_ok)]
print(f"  {len(fft_all):,} rows across {len(fft_files)} subjects after filtering")

# ── Power columns: POW.<ch>.<band> ───────────────────────────────────────────
pow_cols = {}   # (channel, band) -> column name
for ch in CH_LABELS:
    for band in BANDS:
        col = f"POW.{ch}.{band}"
        if col in fft_all.columns:
            pow_cols[(ch, band)] = col

# ── Directed connectivity: SOURCE → each target lobe, per band, per age bin ──
# result[target_lobe][band] = list of values across age bins
#
# For each (i in source, j in target):
#   connectivity += att[i, j] * fft_median[j, band]
# normalised by sum of att[i, j] weights
result = {lobe: {band: [] for band in BANDS} for lobe in LOBES}

for ag_i, ag in enumerate(age_grp_ids):
    grp    = fft_all[fft_all["age_group"] == ag]
    mat    = att_matrices[ag_i]   # [14, 14]: mat[i,j] = i attends to j

    # Precompute median FFT per channel for this age bin
    ch_fft = {}   # channel -> {band -> median_power}
    for ch in CH_LABELS:
        ch_fft[ch] = {}
        for band in BANDS:
            col = pow_cols.get((ch, band))
            ch_fft[ch][band] = grp[col].median() if col else np.nan

    for tgt_lobe in LOBES:
        tgt_idx = lobe_idx[tgt_lobe]
        for band in BANDS:
            weighted_sum = 0.0
            weight_total = 0.0
            for i in src_idx:
                for j in tgt_idx:
                    if i == j:
                        continue
                    w = mat[i, j]   # att from source channel i to target channel j
                    fft_val = ch_fft[CH_LABELS[j]][band]
                    if np.isnan(fft_val):
                        continue
                    weighted_sum += w * fft_val
                    weight_total += w
            result[tgt_lobe][band].append(
                weighted_sum / weight_total if weight_total > 0 else np.nan
            )

# Z-score each (lobe, band) trajectory across age bins
if Z_SCORE:
    for lobe in LOBES:
        for band in BANDS:
            arr = np.array(result[lobe][band], dtype=float)
            std = arr.std()
            if std > 0:
                result[lobe][band] = list((arr - arr.mean()) / std)

# ── Plot (Q2–Q5 only, skip Q1) ────────────────────────────────────────────────
band_titles = {
    "Theta": "Theta  (4–8 Hz)",
    "Alpha": "Alpha  (8–12 Hz)",
    "BetaL": "Beta-L  (12–16 Hz)",
    "BetaH": "Beta-H  (20–28 Hz)",
    "Gamma": "Gamma  (30–100 Hz)",
}

plot_slice = slice(None)             # all bins including Q1
plot_bins  = bin_lbls
x_pos      = np.arange(len(plot_bins))

fig, axes = plt.subplots(1, len(BANDS), figsize=(4.5 * len(BANDS), 4.5), sharey=Z_SCORE)

for ax, band in zip(axes, BANDS):
    lobe_vals = []
    for lobe in LOBES:
        vals = np.array(result[lobe][band])[plot_slice]
        ls = "--" if lobe == SOURCE_LOBE else "-"
        ax.plot(x_pos, vals, marker="o", color=REGION_PAL[lobe],
                label=lobe, linewidth=2.2, markersize=6, linestyle=ls, zorder=3)
        lobe_vals.append(vals)

    # Average across all lobes — dashed black
    avg_vals = np.nanmean(lobe_vals, axis=0)
    ax.plot(x_pos, avg_vals, marker="D", color="black",
            linewidth=2.0, markersize=5, linestyle="--", zorder=4,
            label="Average")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_bins, fontsize=9)
    ax.set_xlabel("Age bin", fontsize=9)
    ax.set_ylabel("Att-weighted power (z-score)" if Z_SCORE else "Att-weighted power", fontsize=9)
    ax.set_title(band_titles[band], fontsize=10, fontweight="bold")
    ax.grid(linewidth=0.4, alpha=0.4, color="0.75")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)

from matplotlib.lines import Line2D
lobe_handles = [Line2D([0],[0], color=REGION_PAL[l], linewidth=2.2,
                        linestyle="--" if l == SOURCE_LOBE else "-",
                        marker="o", markersize=6, label=l) for l in LOBES]
lobe_handles.append(Line2D([0],[0], color="black", linewidth=2.0,
                            linestyle="--", marker="D", markersize=5, label="Average"))
fig.legend(handles=lobe_handles, loc="lower center", ncol=5,
           fontsize=10, frameon=True,
           title=f"Target lobe  (source={SOURCE_LOBE}, dashed=self)",
           title_fontsize=10, bbox_to_anchor=(0.5, -0.08))

head_label = {"HEAD1":"Head 1","HEAD2":"Head 2","HEAD1+HEAD2":"Avg (H1+H2)"}[GRAPH_HEAD]
domain_note = f" | {COG_DOMAIN}" if COG_DOMAIN != "all" else ""


_src_tag  = SOURCE_LOBE.lower()
_head_tag = {"HEAD1":"h1","HEAD2":"h2","HEAD1+HEAD2":"avg"}[GRAPH_HEAD]
_cog_tag  = COG_DOMAIN.lower().replace(" ", "_")
for ext in ("pdf", "png"):
    fname = f"F7_att_weighted_fft_{_src_tag}_{_head_tag}_{_cog_tag}.{ext}"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight")
    print(f"  -> {fname}")
plt.close(fig)
print("Done.")
