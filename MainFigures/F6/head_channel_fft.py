"""
F6_head_channel_fft.py
======================
  Row 1 — lobe attention line plots across age bins:
    A. Head 1  B. Head 2  C. Average (H1+H2)

  Row 2 — SOURCE lobe → target lobe attention-weighted FFT connectivity:
    D. Theta  E. Alpha  F. Beta-L  G. Beta-H  H. Gamma
    Lines = target lobe; X = age bin; Y = att-weighted median FFT power (z-scored)

Config:
  N_BINS      — 3 | 4 | 5  (must match an available attention cache)
  GRAPH_HEAD  — "HEAD1" | "HEAD2" | "HEAD1+HEAD2"
  SOURCE_LOBE — lobe whose outgoing attention is measured in Row 2
  COG_DOMAIN  — "all" | substring of cognitive_domain
  Z_SCORE     — True: z-score row 2 across age bins; False: raw median power

Outputs:
  F6_head_channel_fft_<head>_<source>_<domain>.pdf / .png
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
sys.path.insert(0, _BRAINAGE)

CONFIG_F  = os.path.join(_BRAINAGE, "config.json")
STIM_CSV  = os.path.join(_BRAINAGE, "stimuli_descriptions_and_expectations.csv")
META_XLS  = os.path.join(_BRAINAGE, "data", "PilotDatasetSubjects.xlsx")
RESULTS   = os.path.join(_BRAINAGE, "results")
OUT_DIR   = _SCRIPT_DIR

# ── Config ────────────────────────────────────────────────────────────────────
N_BINS      = 4              # 3 | 4 | 5 — must match an available attention cache
GRAPH_HEAD  = "HEAD1"  # "HEAD1" | "HEAD2" | "HEAD1+HEAD2"
SOURCE_LOBE = "Frontal"      # lobe whose outgoing attention drives Row 2
COG_DOMAIN  = "all"          # "all" | substring of cognitive_domain
Z_SCORE     = True           # z-score Row 2 per (target_lobe, band) across age bins
# ─────────────────────────────────────────────────────────────────────────────

ATT_CACHE = os.path.join(_SCRIPT_DIR, f"attention_cache_{N_BINS}bin.pkl")

BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]
BAND_TITLES = {
    "Theta": "D. Theta  (4–8 Hz)",
    "Alpha": "E. Alpha  (8–12 Hz)",
    "BetaL": "F. Beta-L  (12–16 Hz)",
    "BetaH": "G. Beta-H  (20–28 Hz)",
    "Gamma": "H. Gamma  (30–100 Hz)",
}

REGION_PAL = {
    "Frontal":  "#4C72B0",
    "Temporal": "#DD8452",
    "Parietal": "#55A868",
    "Occipital":"#C44E52",
}
LOBES = ["Frontal", "Temporal", "Parietal", "Occipital"]

_bin_configs = {
    3: [(0,30),(30,50),(50,200)],
    4: [(0,27),(27,34),(34,45),(45,200)],
    5: [(0,25),(25,31),(31,39),(39,47),(47,200)],
}
_known_lbls = {
    3: ["Young\n(<30)",  "Middle\n(30-50)", "Older\n(>50)"],
    4: ["Q1\n(<27)",     "Q2\n(27-33)",     "Q3\n(34-44)",  "Q4\n(>=45)"],
    5: ["Q1\n(<25)",     "Q2\n(25-30)",     "Q3\n(31-38)",  "Q4\n(39-46)", "Q5\n(>=47)"],
}
AGE_GROUPS = [(0,25),(25,31),(31,39),(39,47),(47,200)]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
    "font.size": 10, "font.weight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
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
lobe_idx = {lobe: [i for i, ch in enumerate(CH_LABELS)
                   if REGION_MAP.get(ch) == lobe]
            for lobe in LOBES}

def age_group(age):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(AGE_GROUPS) - 1

# ── Load attention cache ──────────────────────────────────────────────────────
print("Loading attention cache ...")
with open(ATT_CACHE, "rb") as f:
    att = pickle.load(f)

age_grp_ids = sorted({k[1] for k in att})
heads       = sorted({k[2] for k in att})
n_bins      = len(age_grp_ids)
bin_lbls    = _known_lbls.get(n_bins, [f"Bin {ag}" for ag in age_grp_ids])

# ── Stimulus filter ───────────────────────────────────────────────────────────
_stim_df    = pd.read_csv(STIM_CSV)
_BASELINES  = {"eyesopen", "eyesclose"}
_ALLOWED    = set(_stim_df["original_stimuli_name"]) - _BASELINES
_domain_map = dict(zip(_stim_df["original_stimuli_name"],
                       _stim_df["cognitive_domain"].str.lower()))

def _stim_matches(s):
    if s not in _ALLOWED:
        return False
    return True if COG_DOMAIN == "all" else COG_DOMAIN.lower() in _domain_map.get(s, "")

_matched = sorted(s for s in {k[0] for k in att} if _stim_matches(s))
print(f"COG_DOMAIN={COG_DOMAIN!r} → {len(_matched)} stimuli")

# ── Row 1: per-head lobe attention (column-sum, normalised) ──────────────────
att_by_age = {}
for h in heads:
    att_by_age[h] = {}
    for ag in age_grp_ids:
        mats = [v for (s, a, hh), v in att.items() if hh == h and a == ag]
        if not mats:
            continue
        col = np.mean(mats, axis=0).sum(axis=0)
        att_by_age[h][ag] = col / col.sum()

att_by_age_avg = {}
for ag in age_grp_ids:
    if ag in att_by_age[0] and ag in att_by_age[1]:
        att_by_age_avg[ag] = (att_by_age[0][ag] + att_by_age[1][ag]) / 2

def lobe_means(pdata):
    out = {lobe: [] for lobe in LOBES}
    for ag in age_grp_ids:
        if ag not in pdata:
            for lobe in LOBES:
                out[lobe].append(np.nan)
            continue
        for lobe in LOBES:
            idx = lobe_idx[lobe]
            out[lobe].append(pdata[ag][idx].mean() if idx else np.nan)
    return out

lobe_h1  = lobe_means(att_by_age[0])
lobe_h2  = lobe_means(att_by_age[1])
lobe_avg = lobe_means(att_by_age_avg)

# ── Row 2: attention-weighted FFT connectivity ────────────────────────────────
_valid_heads = {"HEAD1", "HEAD2", "HEAD1+HEAD2"}
if GRAPH_HEAD not in _valid_heads:
    raise ValueError(f"GRAPH_HEAD must be one of {_valid_heads}")

# Mean attention matrix per age bin
att_matrices = {}
for ag_i, ag in enumerate(age_grp_ids):
    if GRAPH_HEAD == "HEAD1+HEAD2":
        mats = [v for (s, a, hh), v in att.items() if a == ag and _stim_matches(s)]
    else:
        hh_idx = 0 if GRAPH_HEAD == "HEAD1" else 1
        mats = [v for (s, a, hh), v in att.items()
                if a == ag and hh == hh_idx and _stim_matches(s)]
    att_matrices[ag_i] = np.mean(mats, axis=0) if mats else np.zeros((14, 14))

src_idx = lobe_idx[SOURCE_LOBE]

# Load FFT CSVs
print("Loading FFT data ...")
meta    = pd.read_excel(META_XLS, header=1)[["ID", "Age"]].dropna()
age_map = dict(zip(meta["ID"].astype(str), meta["Age"].astype(float)))

fft_parts = []
for fp in glob.glob(os.path.join(RESULTS, "BA*", "2_emotiv_fft_values.csv")):
    df  = pd.read_csv(fp)
    sid = df["ID"].iloc[0] if "ID" in df.columns else os.path.basename(os.path.dirname(fp))
    age = age_map.get(str(sid))
    if age is None:
        continue
    df["age_group"] = age_group(age)
    fft_parts.append(df)

fft_all = pd.concat(fft_parts, ignore_index=True)
fft_all = fft_all[fft_all["task_name"].apply(_stim_matches)]
print(f"  {len(fft_all):,} FFT rows loaded")

pow_cols = {(ch, band): f"POW.{ch}.{band}"
            for ch in CH_LABELS for band in BANDS
            if f"POW.{ch}.{band}" in fft_all.columns}

# Directed connectivity: SOURCE → each target lobe, per band, per age bin
conn = {lobe: {band: [] for band in BANDS} for lobe in LOBES}

for ag_i, ag in enumerate(age_grp_ids):
    grp = fft_all[fft_all["age_group"] == ag]
    mat = att_matrices[ag_i]

    ch_fft = {ch: {band: grp[pow_cols[(ch, band)]].median()
                   if (ch, band) in pow_cols else np.nan
                   for band in BANDS}
              for ch in CH_LABELS}

    for tgt_lobe in LOBES:
        tgt_idx = lobe_idx[tgt_lobe]
        for band in BANDS:
            wsum, wtot = 0.0, 0.0
            for i in src_idx:
                for j in tgt_idx:
                    if i == j:
                        continue
                    w = mat[i, j]
                    v = ch_fft[CH_LABELS[j]][band]
                    if not np.isnan(v):
                        wsum += w * v
                        wtot += w
            conn[tgt_lobe][band].append(wsum / wtot if wtot > 0 else np.nan)

if Z_SCORE:
    for lobe in LOBES:
        for band in BANDS:
            arr = np.array(conn[lobe][band], dtype=float)
            std = arr.std()
            if std > 0:
                conn[lobe][band] = list((arr - arr.mean()) / std)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
outer  = fig.add_gridspec(2, 1, hspace=0.65, height_ratios=[1, 1.7])
gs_top = outer[0].subgridspec(1, 3, wspace=0.38)
gs_bot = outer[1].subgridspec(1, 5, wspace=0.10)

def style_ax(ax):
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(1.0)

x_pos = np.arange(n_bins)

# ── Row 1: A, B, C ───────────────────────────────────────────────────────────
for pi, (lbl, ldata, title) in enumerate([
    ("A", lobe_h1,  "Head 1 - mean attention per lobe"),
    ("B", lobe_h2,  "Head 2 - mean attention per lobe"),
    ("C", lobe_avg, "Avg (H1+H2) - attention per lobe"),
]):
    ax = fig.add_subplot(gs_top[pi])
    for lobe in LOBES:
        ax.plot(x_pos, ldata[lobe], marker="o", color=REGION_PAL[lobe],
                label=lobe, linewidth=2.2, markersize=6, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_lbls, fontsize=9)
    ax.set_xlabel("Age bin", fontsize=9)
    ax.set_ylabel("Mean normalised attention", fontsize=9)
    ax.set_title(f"{lbl}. {title}", fontsize=10, fontweight="bold", loc="left")
    ax.grid(linewidth=0.4, alpha=0.4, color="0.75")
    ax.set_axisbelow(True)
    style_ax(ax)

# ── Row 2: D–H (one per FFT band) — shared y-axis ───────────────────────────
bot_axes = []
for bi in range(len(BANDS)):
    ax = fig.add_subplot(gs_bot[bi], sharey=bot_axes[0] if bi > 0 else None)
    bot_axes.append(ax)

for bi, (band, ax) in enumerate(zip(BANDS, bot_axes)):
    for lobe in LOBES:
        ls = "--" if lobe == SOURCE_LOBE else "-"
        ax.plot(x_pos, conn[lobe][band], marker="o", color=REGION_PAL[lobe],
                linewidth=2.2, markersize=6, linestyle=ls, zorder=3)
    avg_vals = np.nanmean([conn[l][band] for l in LOBES], axis=0)
    ax.plot(x_pos, avg_vals, marker="D", color="black",
            linewidth=2.0, markersize=5, linestyle="--", zorder=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_lbls, fontsize=9)
    ax.set_xlabel("Age bin", fontsize=9)
    if bi == 0:
        ax.set_ylabel("Att-weighted power (z-score)" if Z_SCORE else "Att-weighted power",
                      fontsize=9)
    else:
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(BAND_TITLES[band], fontsize=10, fontweight="bold", loc="left")
    ax.grid(linewidth=0.4, alpha=0.4, color="0.75")
    ax.set_axisbelow(True)
    style_ax(ax)

# ── Shared legend ─────────────────────────────────────────────────────────────
lobe_handles = [Line2D([0],[0], color=REGION_PAL[l], linewidth=2.2,
                        linestyle="--" if l == SOURCE_LOBE else "-",
                        marker="o", markersize=6, label=l) for l in LOBES]
lobe_handles.append(Line2D([0],[0], color="black", linewidth=2.0,
                            linestyle="--", marker="D", markersize=5, label="Average"))
fig.legend(handles=lobe_handles, loc="lower center", ncol=5,
           fontsize=11, frameon=True,
           title_fontsize=11, bbox_to_anchor=(0.5, -0.12))

_head_tag = {"HEAD1":"h1","HEAD2":"h2","HEAD1+HEAD2":"avg"}[GRAPH_HEAD]
_src_tag  = SOURCE_LOBE.lower()
_cog_tag  = COG_DOMAIN.lower().replace(" ", "_")
for ext in ("pdf", "png"):
    fname = f"F6_head_channel_fft_{_head_tag}_{_src_tag}_{_cog_tag}.{ext}"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight")
    print(f"  -> {fname}")
plt.close(fig)
print("Done.")
