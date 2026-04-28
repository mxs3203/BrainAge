"""
F7_lobe_connectivity.py
=======================
Attention-weighted FFT connectivity: SOURCE lobe -> each target lobe.

Panels:
  columns = FFT bands (Theta | Alpha | BetaL | BetaH | Gamma)
  rows    = cognitive domains listed in COG_DOMAINS

Outputs:
  F7_att_weighted_fft_<source>_<head>_cog_domains_grid.pdf / .png
"""
import os, sys, json, pickle, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAINAGE   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_F6_DIR     = os.path.join(_SCRIPT_DIR, "..", "F6")
sys.path.insert(0, _BRAINAGE)

N_BINS     = 4
ATT_CACHE  = os.path.join(_F6_DIR, f"attention_cache_{N_BINS}bin.pkl")
CONFIG_F   = os.path.join(_BRAINAGE, "config.json")
STIM_CSV   = os.path.join(_BRAINAGE, "stimuli_descriptions_and_expectations.csv")
META_XLS   = os.path.join(_BRAINAGE, "data", "PilotDatasetSubjects.xlsx")
RESULTS    = os.path.join(_BRAINAGE, "results")
OUT_DIR    = _SCRIPT_DIR

SOURCE_LOBE = "Frontal"
GRAPH_HEAD  = "HEAD1"        # "HEAD1" | "HEAD2" | "HEAD1+HEAD2"
COG_DOMAINS = [
    # "Executive reasoning",
    # "Speech production",
    # "Memorization",
    # "Recall",
    # "Spatial attention",
    "Motor coordination",
    "Pattern recognition",
    "Auditory perception",
    "Emotion recognition",
]
Z_SCORE = True

LOBES = ["Frontal", "Temporal", "Parietal", "Occipital"]
BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]
REGION_PAL = {
    "Frontal": "#4C72B0",
    "Temporal": "#DD8452",
    "Parietal": "#55A868",
    "Occipital": "#C44E52",
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
CH_LABELS = [c.replace("EEG.", "") for c in CHANNELS]

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
    3: ["Young\n(<30)", "Middle\n(30-50)", "Older\n(>50)"],
    4: ["Q1\n(<27)", "Q2\n(27-33)", "Q3\n(34-44)", "Q4\n(>=45)"],
    5: ["Q1\n(<25)", "Q2\n(25-30)", "Q3\n(31-38)", "Q4\n(39-46)", "Q5\n(>=47)"],
}

def age_group(age):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(AGE_GROUPS) - 1

_stim_df = pd.read_csv(STIM_CSV)
_stim_df["cognitive_domain_norm"] = _stim_df["cognitive_domain"].fillna("").str.lower()
_BASELINES = {"eyesopen", "eyesclose"}
_ALLOWED = set(_stim_df["original_stimuli_name"]) - _BASELINES
_domain_map = dict(zip(_stim_df["original_stimuli_name"], _stim_df["cognitive_domain_norm"]))

def stim_ok(s, cog_domain):
    if s not in _ALLOWED:
        return False
    if cog_domain == "all":
        return True
    return cog_domain.lower() in _domain_map.get(s, "")

print("Loading attention cache ...")
with open(ATT_CACHE, "rb") as f:
    att = pickle.load(f)

age_grp_ids = sorted({k[1] for k in att})
n_bins = len(age_grp_ids)
bin_lbls = _known_lbls.get(n_bins, [f"Bin {ag}" for ag in age_grp_ids])

_valid_heads = {"HEAD1", "HEAD2", "HEAD1+HEAD2"}
if GRAPH_HEAD not in _valid_heads:
    raise ValueError(f"GRAPH_HEAD must be one of {_valid_heads}")

src_idx = lobe_idx[SOURCE_LOBE]

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
    df["age"] = age
    fft_parts.append(df)

if not fft_parts:
    raise RuntimeError("No FFT files were loaded. Check RESULTS and META_XLS paths.")

fft_all = pd.concat(fft_parts, ignore_index=True)
fft_all = fft_all[fft_all["task_name"].isin(_ALLOWED)].copy()
print(f"  {len(fft_all):,} baseline filtered rows across {len(fft_files)} files")

pow_cols = {}
for ch in CH_LABELS:
    for band in BANDS:
        col = f"POW.{ch}.{band}"
        if col in fft_all.columns:
            pow_cols[(ch, band)] = col

def build_attention_matrices(cog_domain):
    att_matrices = {}
    for ag_i, ag in enumerate(age_grp_ids):
        if GRAPH_HEAD == "HEAD1+HEAD2":
            mats = [v for (s, a, hh), v in att.items() if a == ag and stim_ok(s, cog_domain)]
        else:
            hh_idx = 0 if GRAPH_HEAD == "HEAD1" else 1
            mats = [v for (s, a, hh), v in att.items()
                    if a == ag and hh == hh_idx and stim_ok(s, cog_domain)]
        att_matrices[ag_i] = np.mean(mats, axis=0) if mats else np.full((14, 14), np.nan)
    return att_matrices

def compute_domain_result(cog_domain):
    domain_fft = fft_all[fft_all["task_name"].apply(lambda s: stim_ok(s, cog_domain))]
    att_matrices = build_attention_matrices(cog_domain)
    result = {lobe: {band: [] for band in BANDS} for lobe in LOBES}

    for ag_i, ag in enumerate(age_grp_ids):
        grp = domain_fft[domain_fft["age_group"] == ag]
        mat = att_matrices[ag_i]

        ch_fft = {}
        for ch in CH_LABELS:
            ch_fft[ch] = {}
            for band in BANDS:
                col = pow_cols.get((ch, band))
                ch_fft[ch][band] = grp[col].median() if col and len(grp) else np.nan

        for tgt_lobe in LOBES:
            tgt_idx = lobe_idx[tgt_lobe]
            for band in BANDS:
                weighted_sum = 0.0
                weight_total = 0.0
                for i in src_idx:
                    for j in tgt_idx:
                        if i == j:
                            continue
                        w = mat[i, j]
                        fft_val = ch_fft[CH_LABELS[j]][band]
                        if np.isnan(w) or np.isnan(fft_val):
                            continue
                        weighted_sum += w * fft_val
                        weight_total += w
                result[tgt_lobe][band].append(
                    weighted_sum / weight_total if weight_total > 0 else np.nan
                )

    if Z_SCORE:
        for lobe in LOBES:
            for band in BANDS:
                arr = np.array(result[lobe][band], dtype=float)
                std = np.nanstd(arr)
                if std > 0:
                    result[lobe][band] = list((arr - np.nanmean(arr)) / std)

    return result, len(domain_fft)

band_titles = {
    "Theta": "Theta  (4-8 Hz)",
    "Alpha": "Alpha  (8-12 Hz)",
    "BetaL": "Beta-L  (12-16 Hz)",
    "BetaH": "Beta-H  (20-28 Hz)",
    "Gamma": "Gamma  (30-100 Hz)",
}

plot_bins = bin_lbls
x_pos = np.arange(len(plot_bins))
n_rows = len(COG_DOMAINS)
n_cols = len(BANDS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.2 * n_cols, max(3.0 * n_rows, 7)),
    sharex=True,
    sharey=Z_SCORE,
    squeeze=False,
)

for row_i, cog_domain in enumerate(COG_DOMAINS):
    print(f"Computing domain: {cog_domain}")
    result, n_domain_rows = compute_domain_result(cog_domain)

    for col_i, band in enumerate(BANDS):
        ax = axes[row_i, col_i]
        lobe_vals = []
        for lobe in LOBES:
            vals = np.array(result[lobe][band], dtype=float)
            ls = "--" if lobe == SOURCE_LOBE else "-"
            ax.plot(x_pos, vals, marker="o", color=REGION_PAL[lobe],
                    label=lobe, linewidth=2.0, markersize=5,
                    linestyle=ls, zorder=3)
            lobe_vals.append(vals)

        with np.errstate(all="ignore"):
            avg_vals = np.nanmean(lobe_vals, axis=0)
        ax.plot(x_pos, avg_vals, marker="D", color="black",
                linewidth=1.8, markersize=4.5, linestyle="--",
                zorder=4, label="Average")

        if row_i == 0:
            ax.set_title(band_titles[band], fontsize=10, fontweight="bold")
        if col_i == 0:
            ylabel = "Att weighted power\n(z score)" if Z_SCORE else "Att weighted power"
            ax.set_ylabel(f"{cog_domain}\n{ylabel}", fontsize=9)
        if row_i == n_rows - 1:
            ax.set_xlabel("Age bin", fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_bins, fontsize=8)
        ax.grid(linewidth=0.4, alpha=0.4, color="0.75")
        ax.set_axisbelow(True)
        ax.text(0.02, 0.95, f"n rows={n_domain_rows:,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8, pad=2))
        for sp in ax.spines.values():
            sp.set_color("black")
            sp.set_linewidth(0.8)

lobe_handles = [
    Line2D([0], [0], color=REGION_PAL[l], linewidth=2.2,
           linestyle="--" if l == SOURCE_LOBE else "-",
           marker="o", markersize=6, label=l)
    for l in LOBES
]
lobe_handles.append(Line2D([0], [0], color="black", linewidth=2.0,
                           linestyle="--", marker="D", markersize=5,
                           label="Average"))

head_label = {"HEAD1":"Head 1", "HEAD2":"Head 2", "HEAD1+HEAD2":"Avg (H1+H2)"}[GRAPH_HEAD]
fig.suptitle(f"Attention weighted FFT connectivity | source={SOURCE_LOBE} | {head_label}",
             fontsize=13, fontweight="bold", y=0.995)
fig.legend(handles=lobe_handles, loc="lower center", ncol=5,
           fontsize=10, frameon=True,
           title=f"Target lobe  (source={SOURCE_LOBE}, dashed=self)",
           title_fontsize=10, bbox_to_anchor=(0.5, 0.002))
fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.975])

_src_tag = SOURCE_LOBE.lower()
_head_tag = {"HEAD1":"h1", "HEAD2":"h2", "HEAD1+HEAD2":"avg"}[GRAPH_HEAD]
for ext in ("pdf", "png"):
    fname = f"F7_att_weighted_fft_{_src_tag}_{_head_tag}_cog_domains_grid.{ext}"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight")
    print(f"  -> {fname}")
plt.close(fig)
print("Done.")
