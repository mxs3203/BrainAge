"""
umap_all_channels.py — UMAP embeddings for all 14 EEG channel encoders.

Loads N_SAMPLES random windows, runs each channel through its pretrained
EEGConvGRUEncoder, fits UMAP, computes quantitative metrics, and saves:

  umap_embeddings_cache.pkl      cached embeddings + UMAP coords (reused on re-run)
  SF6_umap_all_channels.pdf/.png Supplementary Figure 6: 14-panel UMAP grid
  F4_umap_main.pdf/.png          Main figure: 2 best-channel UMAPs + metrics bars
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
import umap

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAINAGE   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_MODELING   = os.path.join(_BRAINAGE, "Modeling_ContrastiveChannelWise")
sys.path.insert(0, _BRAINAGE)
os.chdir(_MODELING)

from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder
from utils.weighted_sampling_age import load_or_create_index

# ── Config ────────────────────────────────────────────────────────────────────
CHKPT_DIR    = "ContrastiveModel_chkpoints"
ROOT_DIR     = "../results/EEG_LSTM_windows/"
PKL_PATH     = "eeg_window_index.pkl"
HYPERPARAM_F = "contrastive_hyperparams.json"
CONFIG_F     = "../config.json"
CACHE_FILE   = os.path.join(_SCRIPT_DIR, "umap_embeddings_cache.pkl")
OUT_DIR      = _SCRIPT_DIR

N_SAMPLES    = 5000
RANDOM_STATE = 42
BATCH_SIZE   = 2048
NUM_WORKERS  = 8
KNN_K        = 10          # for kNN-MAE metric
UMAP_KWARGS  = dict(n_neighbors=30, min_dist=0.1, metric="cosine",
                    random_state=RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_F) as f:
    CHANNELS = json.load(f)["EEG_columns"]
CH_LABELS = [c.replace("EEG.", "") for c in CHANNELS]

with open(HYPERPARAM_F) as f:
    HYPERPARAMS = json.load(f)

REGION_MAP = {
    "AF3": "Frontal",  "F7": "Frontal",   "F3": "Frontal",  "FC5": "Frontal",
    "T7":  "Temporal", "P7": "Parietal",  "O1": "Occipital","O2":  "Occipital",
    "P8":  "Parietal", "T8": "Temporal",  "FC6":"Frontal",  "F4":  "Frontal",
    "F8":  "Frontal",  "AF4":"Frontal",
}
REGION_PALETTE = {
    "Frontal": "#4C72B0", "Temporal": "#DD8452",
    "Parietal": "#55A868", "Occipital": "#C44E52",
}

AGE_CMAP = "bwr"          # blue = young, red = old


# ── Dataset ───────────────────────────────────────────────────────────────────
class EEGWindowDataset(Dataset):
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            with open(self.paths[idx], "rb") as fh:
                s = pickle.load(fh)
            data = s.get("X", s.get("x"))
            if data is None:
                return None
            X   = torch.tensor(data, dtype=torch.float32)   # [512, 14]
            age = float(s.get("Age", float("nan")))
            return X, age
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    Xs, ages = zip(*batch)
    return torch.stack(Xs), torch.tensor(ages, dtype=torch.float32)


# ── Encoder helpers ───────────────────────────────────────────────────────────
def load_encoder(channel_name):
    hp  = HYPERPARAMS[channel_name]
    enc = EEGConvGRUEncoder(
        gru_hidden     = hp["hidden_size"],
        embedding_dim  = hp["projection_size"],
        num_layers_gru = hp["num_layers_gru"],
        kernel_size    = hp["kernel_size"],
        stride         = hp["stride"],
        padding        = hp["padding"],
        cnn_dropout    = hp["CNN_dropout"],
        activation     = hp["activation_function"],
        conv_channels  = hp["Conv1DChannels"],
        rnn_dropout    = hp["RNN_dropout"],
        rnn_type       = hp["rnn_type"],
    )
    ckpt = os.path.join(CHKPT_DIR, f"EEGContrsative_forCH_{channel_name}.pth")
    sd   = torch.load(ckpt, map_location=DEVICE)
    sd   = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd)
    return enc.to(DEVICE).eval()


def encode_channel(encoder, all_X, ch_idx):
    """Run pre-loaded tensor all_X [N, 512, 14] through encoder for one channel."""
    embs = []
    with torch.no_grad():
        for start in range(0, all_X.shape[0], BATCH_SIZE):
            chunk = all_X[start:start + BATCH_SIZE, :, ch_idx].unsqueeze(-1).to(DEVICE)
            emb, _ = encoder(chunk)
            embs.append(emb.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


# ── Quantitative metrics ──────────────────────────────────────────────────────
def compute_metrics(Z, ages, umap_2d):
    r1 = abs(float(spearmanr(umap_2d[:, 0], ages)[0]))
    r2 = abs(float(spearmanr(umap_2d[:, 1], ages)[0]))
    best_r = max(r1, r2)

    knn = KNeighborsRegressor(n_neighbors=KNN_K, metric="cosine")
    knn.fit(Z, ages)
    knn_mae = float(np.abs(knn.predict(Z) - ages).mean())

    # Alignment: mean squared L2 for same-age (±2 yr) random pairs
    rng   = np.random.default_rng(0)
    n     = len(Z)
    ia, ib = rng.integers(0, n, 8000), rng.integers(0, n, 8000)
    close = np.abs(ages[ia] - ages[ib]) < 2.0
    if close.sum() > 5:
        diffs = Z[ia[close]] - Z[ib[close]]
        align = float(np.mean(np.sum(diffs ** 2, axis=1)))
    else:
        align = float("nan")

    # Uniformity: log-mean-exp(-2||zi-zj||^2) on a random subset
    sub  = rng.integers(0, n, min(2000, n))
    Zs   = Z[sub]
    sqpd = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=-1)
    unif = float(np.log(np.mean(np.exp(-2 * sqpd)) + 1e-12))

    return {
        "spearman_r":  best_r,
        "knn_mae":     knn_mae,
        "alignment":   align,
        "uniformity":  unif,
    }


# ── Collect or load cache ──────────────────────────────────────────────────────
def collect_or_load():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cache from {CACHE_FILE} …")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Building UMAP cache …")
    index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
    rng      = np.random.default_rng(RANDOM_STATE)
    idx_sel  = rng.choice(len(index_df), min(N_SAMPLES, len(index_df)), replace=False)
    paths    = index_df.iloc[idx_sel]["file_path"].tolist()

    # Load all N windows into CPU RAM once
    print(f"Loading {len(paths)} windows …")
    loader = DataLoader(
        EEGWindowDataset(paths),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        persistent_workers=True, collate_fn=collate_fn,
    )
    X_list, age_list = [], []
    for Xb, ab in tqdm(loader, desc="Reading windows"):
        if Xb is None:
            continue
        X_list.append(Xb)
        age_list.append(ab)
    all_X   = torch.cat(X_list, dim=0)        # [N, 512, 14]
    all_ages = torch.cat(age_list).numpy()     # [N]

    # Filter NaN ages
    valid   = np.isfinite(all_ages)
    all_X   = all_X[valid]
    all_ages = all_ages[valid]
    print(f"  {valid.sum()} valid samples after NaN filter")

    cache = {}
    for ch_name, ch_label in zip(CHANNELS, CH_LABELS):
        print(f"\nChannel {ch_label} …")
        enc     = load_encoder(ch_name)
        ch_idx  = CHANNELS.index(ch_name)
        Z       = encode_channel(enc, all_X, ch_idx)
        del enc; torch.cuda.empty_cache()

        print(f"  UMAP …")
        reducer = umap.UMAP(**UMAP_KWARGS)
        umap_2d = reducer.fit_transform(Z)

        metrics = compute_metrics(Z, all_ages, umap_2d)
        print(f"  spearman_r={metrics['spearman_r']:.3f}  "
              f"knn_mae={metrics['knn_mae']:.2f}  "
              f"align={metrics['alignment']:.4f}  "
              f"unif={metrics['uniformity']:.3f}")

        cache[ch_name] = {
            "Z":       Z,
            "ages":    all_ages,
            "umap_2d": umap_2d,
            "metrics": metrics,
            "label":   ch_label,
        }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved cache → {CACHE_FILE}")
    return cache


# ── Supplementary Figure S6 ──────────────────────────────────────────────────
def make_s6(cache):
    """3-column × 5-row grid of all 14 channel UMAPs."""
    n_cols = 3
    n_rows = 5   # 15 slots; last one hidden
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 20))
    axes = axes.flatten()

    age_all = np.concatenate([cache[ch]["ages"] for ch in CHANNELS])
    vmin, vmax = np.percentile(age_all, 2), np.percentile(age_all, 98)

    for i, (ch_name, ch_label) in enumerate(zip(CHANNELS, CH_LABELS)):
        ax  = axes[i]
        d   = cache[ch_name]
        xy  = d["umap_2d"]
        ages = d["ages"]
        region = REGION_MAP.get(ch_label, "Frontal")

        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ages, cmap=AGE_CMAP,
                        vmin=vmin, vmax=vmax, s=6, alpha=0.75, edgecolors="none",
                        rasterized=True)
        ax.set_title(
            f"{ch_label}  [{region[0]}]",
            fontsize=10, fontweight="bold",
            color=REGION_PALETTE[region],
        )
        ax.set_xlabel("UMAP-1", fontsize=7)
        ax.set_ylabel("UMAP-2", fontsize=7)
        ax.tick_params(labelsize=6)

        m = d["metrics"]
        ax.text(0.03, 0.97,
                f"r={m['spearman_r']:.2f}  MAE={m['knn_mae']:.1f}",
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))

    # Shared colorbar
    fig.colorbar(sc, ax=axes[:len(CHANNELS)], shrink=0.4, pad=0.02, label="Age (years)")

    # Hide leftover panel
    for j in range(len(CHANNELS), len(axes)):
        axes[j].set_visible(False)

    # Region legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           markersize=9, label=r)
               for r, c in REGION_PALETTE.items()]
    fig.legend(handles=handles, loc="lower right", frameon=True,
               fontsize=9, title="Region", title_fontsize=9)

    fig.suptitle("UMAP embeddings — channel-wise contrastive encoders", fontsize=14,
                 fontweight="bold", y=1.002)
    fig.tight_layout()
    return fig


# ── Main Figure F4 ───────────────────────────────────────────────────────────
MAIN_CHANNELS = ["EEG.F4", "EEG.AF4"]   # channels shown as example UMAPs in main figure


def make_f4(cache):
    """F4 and AF4 UMAPs (top row) + kNN-MAE and Spearman-r bars (bottom row)."""

    best2 = MAIN_CHANNELS

    age_all = np.concatenate([cache[ch]["ages"] for ch in CHANNELS])
    vmin, vmax = np.percentile(age_all, 2), np.percentile(age_all, 98)

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.32,
                            height_ratios=[1.1, 0.9])

    # ── Top row: 2 UMAPs ─────────────────────────────────────────────────────
    for col, ch_name in enumerate(best2):
        ax  = fig.add_subplot(gs[0, col])
        d   = cache[ch_name]
        xy  = d["umap_2d"]
        ages = d["ages"]
        region = REGION_MAP.get(d["label"], "Frontal")

        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ages, cmap=AGE_CMAP,
                        vmin=vmin, vmax=vmax, s=8, alpha=0.80,
                        edgecolors="none", rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Age (years)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        m = d["metrics"]
        panel_lbl = "A" if col == 0 else "B"
        ax.set_title(
            f"{panel_lbl}. {d['label']}  [{region}]\n"
            f"kNN-MAE = {m['knn_mae']:.2f} yr  |r| = {m['spearman_r']:.3f}",
            fontsize=10, fontweight="bold", loc="left",
        )
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Top-right: blank or summary stats table
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    rows = []
    for ch_name in CHANNELS:
        d  = cache[ch_name]
        m  = d["metrics"]
        rows.append([d["label"],
                     f"{m['knn_mae']:.2f}",
                     f"{m['spearman_r']:.3f}",
                     f"{m['alignment']:.4f}",
                     f"{m['uniformity']:.2f}"])
    rows.sort(key=lambda r: float(r[1]))   # sort by kNN-MAE

    col_labels = ["Channel", "kNN-MAE", "|r|", "Align", "Unif"]
    tbl = ax_info.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    # Bold header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#DDEEFF")
        tbl[(0, j)].set_text_props(fontweight="bold")
    # Highlight the two example channels (F4 and AF4)
    for row_i, row in enumerate(rows):
        ch_label_row = row[0]
        if ch_label_row in [cache[c]["label"] for c in MAIN_CHANNELS]:
            for j in range(len(col_labels)):
                tbl[(row_i + 1, j)].set_facecolor("#D5F5D5")
    ax_info.set_title("C. Embedding metrics — all channels\n(sorted by kNN-MAE ↑)",
                      fontsize=9, fontweight="bold", loc="left")

    # ── Bottom row: metric bar charts ─────────────────────────────────────────
    # Sort channels by kNN-MAE for bars
    ch_sorted  = sorted(CHANNELS, key=lambda c: cache[c]["metrics"]["knn_mae"])
    lbl_sorted = [cache[c]["label"] for c in ch_sorted]
    colors_bar = [REGION_PALETTE[REGION_MAP.get(l, "Frontal")] for l in lbl_sorted]

    # Panel D — kNN-MAE
    ax_mae = fig.add_subplot(gs[1, :2])
    knn_vals = [cache[c]["metrics"]["knn_mae"] for c in ch_sorted]
    main_labels = {cache[c]["label"] for c in MAIN_CHANNELS}
    bars = ax_mae.bar(lbl_sorted, knn_vals, color=colors_bar, edgecolor="white",
                      linewidth=0.5, zorder=3)
    for bi, lbl in enumerate(lbl_sorted):
        if lbl in main_labels:
            bars[bi].set_edgecolor("black")
            bars[bi].set_linewidth(1.8)
    ax_mae.set_ylabel("kNN-MAE (years)", fontsize=9)
    ax_mae.set_xlabel("Channel", fontsize=9)
    ax_mae.set_title("D. Age prediction accuracy from embeddings  (kNN, k=10; lower = better)",
                     fontsize=9, fontweight="bold", loc="left")
    ax_mae.tick_params(axis="x", rotation=45, labelsize=8)
    ax_mae.tick_params(axis="y", labelsize=8)
    ax_mae.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_mae.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax_mae.set_axisbelow(True)

    # Panel E — Spearman |r|
    ax_r = fig.add_subplot(gs[1, 2])
    r_vals   = [cache[c]["metrics"]["spearman_r"] for c in ch_sorted]
    bars2 = ax_r.barh(lbl_sorted, r_vals, color=colors_bar, edgecolor="white",
                       linewidth=0.5, zorder=3)
    for bi, lbl in enumerate(lbl_sorted):
        if lbl in main_labels:
            bars2[bi].set_edgecolor("black")
            bars2[bi].set_linewidth(1.8)
    ax_r.set_xlabel("|Spearman r|  (age vs. UMAP axis)", fontsize=8)
    ax_r.set_title("E. Age–UMAP\ncorrelation",
                   fontsize=9, fontweight="bold", loc="left")
    ax_r.tick_params(labelsize=8)
    ax_r.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_r.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    ax_r.set_axisbelow(True)

    # Region legend
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=c, markersize=9, label=r)
               for r, c in REGION_PALETTE.items()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, -0.01),
               frameon=True, fontsize=8, title="Region", title_fontsize=8,
               ncol=4)

    return fig, best2


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cache = collect_or_load()

    print("\nGenerating Supplementary Figure S6 …")
    fig_s6 = make_s6(cache)
    fig_s6.savefig(os.path.join(OUT_DIR, "SF6_umap_all_channels.pdf"),
                   bbox_inches="tight")
    fig_s6.savefig(os.path.join(OUT_DIR, "SF6_umap_all_channels.png"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig_s6)
    print("  → SF6_umap_all_channels.pdf/png")

    print("Generating Main Figure F4 (UMAPs) …")
    fig_f4, best2 = make_f4(cache)
    fig_f4.savefig(os.path.join(OUT_DIR, "F4_umap_main.pdf"),
                   bbox_inches="tight")
    fig_f4.savefig(os.path.join(OUT_DIR, "F4_umap_main.png"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig_f4)
    best_labels = [cache[c]["label"] for c in best2]
    print(f"  → F4_umap_main.pdf/png  (best channels: {best_labels})")

    print("\nMetrics summary:")
    for ch in sorted(CHANNELS, key=lambda c: cache[c]["metrics"]["knn_mae"]):
        m = cache[ch]["metrics"]
        print(f"  {cache[ch]['label']:5s}  knn_mae={m['knn_mae']:.2f}  "
              f"|r|={m['spearman_r']:.3f}  align={m['alignment']:.4f}  "
              f"unif={m['uniformity']:.2f}")
