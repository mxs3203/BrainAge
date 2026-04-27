"""
Figure 6 — Cross-channel attention weights by age group and stimulus.

Runs the full dataset through the AgeModel, collects the MultiheadAttention
weights [B, num_heads, 14, 14], and accumulates per-stimulus × age-group means.
Results are cached to ATT_CACHE so the forward pass only runs once.

Visualisation: for each stimulus one figure with rows = age groups,
columns = attention heads, showing the 14×14 channel-channel heatmap.
A summary figure averages over stimuli.
"""
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAINAGE    = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_MODELING    = os.path.join(_BRAINAGE, "Modeling_ContrastiveChannelWise")
sys.path.insert(0, _BRAINAGE)
os.chdir(_MODELING)   # relative paths inside AgeModel expect this cwd

from ModelDefs.AttAgeModel import AgeModel
from utils.weighted_sampling_age import load_or_create_index, age_to_bin

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "ContrastiveModel_chkpoints/FullModel_Epoch_v4_46_bn_calibrated.pth"
ROOT_DIR    = "../results/EEG_LSTM_windows/"
PKL_PATH    = "eeg_window_index.pkl"
ATT_CACHE   = os.path.join(_SCRIPT_DIR, "attention_cache.pkl")
OUT_DIR     = _SCRIPT_DIR

BATCH_SIZE  = 10000
NUM_WORKERS = 32

ATT_HEADS   = 2
COMMON_DIM  = 16
FC1_DIM     = 512
FC2_DIM     = 512
ACTIVATION  = "SiLU"
DROPOUT     = 0.1

with open(os.path.join(_BRAINAGE, "config.json")) as f:
    CHANNELS = json.load(f)["EEG_columns"]
CH_LABELS = [c.replace("EEG.", "") for c in CHANNELS]   # strip "EEG." prefix

# Age group boundaries (years)
AGE_GROUPS   = [(0, 30), (30, 50), (50, 200)]
AGE_LABELS   = ["Young (<30)", "Middle (30–50)", "Older (>50)"]
AGE_SHORT    = ["young", "middle", "old"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class EEGAttDataset(Dataset):
    def __init__(self, index_df):
        self.df = index_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            with open(row["file_path"], "rb") as fh:
                s = pickle.load(fh)
            data = s.get("X", s.get("x"))
            if data is None:
                return None
            X       = torch.tensor(data, dtype=torch.float32)
            age     = float(s.get("Age", float("nan")))
            stim    = str(s.get("stimulus", ""))
            return X, age, stim
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    Xs, ages, stims = zip(*batch)
    return torch.stack(Xs), list(ages), list(stims)


def age_group(age):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age < hi:
            return i
    return len(AGE_GROUPS) - 1


# ── Collect or load cache ──────────────────────────────────────────────────────
def collect_attention():
    """Return dict: (stimulus, age_group_idx, head_idx) -> mean att matrix [14,14]."""
    if os.path.exists(ATT_CACHE):
        print(f"Loading cached attention weights from {ATT_CACHE} …")
        with open(ATT_CACHE, "rb") as f:
            return pickle.load(f)

    print("No cache found — running full dataset forward pass …")

    # Load model
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = AgeModel(
        attention_heads=ATT_HEADS, common_dim=COMMON_DIM,
        fc1_dim=FC1_DIM, fc2_dim=FC2_DIM, activation=ACTIVATION,
        dropout=DROPOUT, unlock_encoders=False,
        hyperparam_config="contrastive_hyperparams.json",
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()

    # Load dataset
    index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
    loader = DataLoader(
        EEGAttDataset(index_df),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn,
    )

    # Accumulators: (stim, age_grp, head) -> [sum_matrix, count]
    acc = {}  # key -> [np.ndarray(14,14), int]

    with torch.no_grad():
        for X_batch, ages, stims in tqdm(loader, desc="Collecting attention"):
            if X_batch is None:
                continue
            X_batch = X_batch.to(DEVICE)
            _, att = model(X_batch)
            # att: [B, num_heads, 14, 14]  (average_attn_weights=False)
            att_np = att.cpu().float().numpy()

            for b in range(len(ages)):
                ag  = age_group(ages[b])
                stm = stims[b]
                for h in range(ATT_HEADS):
                    key = (stm, ag, h)
                    mat = att_np[b, h]          # [14, 14]
                    if key not in acc:
                        acc[key] = [np.zeros((14, 14)), 0]
                    acc[key][0] += mat
                    acc[key][1] += 1

    # Compute means
    result = {k: v[0] / v[1] for k, v in acc.items()}

    os.makedirs(os.path.dirname(ATT_CACHE) or ".", exist_ok=True)
    with open(ATT_CACHE, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved attention cache → {ATT_CACHE}")
    return result


# ── Plotting helpers ───────────────────────────────────────────────────────────
_CMAP = "viridis"

def plot_heatmap(ax, mat, title, vmin, vmax, show_cbar=False):
    im = ax.imshow(mat, cmap=_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(14))
    ax.set_yticks(range(14))
    ax.set_xticklabels(CH_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(CH_LABELS, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    if show_cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def make_stimulus_figure(att_dict, stim):
    """One figure per stimulus: rows=age groups, cols=heads."""
    n_groups = len(AGE_GROUPS)
    fig, axes = plt.subplots(n_groups, ATT_HEADS,
                             figsize=(5 * ATT_HEADS, 4.5 * n_groups),
                             squeeze=False)
    fig.suptitle(f"Cross-channel attention — {stim}", fontsize=12, fontweight="bold")

    # Determine shared colour scale across all panels for this stimulus
    all_vals = []
    for ag in range(n_groups):
        for h in range(ATT_HEADS):
            key = (stim, ag, h)
            if key in att_dict:
                all_vals.append(att_dict[key])
    if not all_vals:
        plt.close(fig)
        return None
    vmin = min(m.min() for m in all_vals)
    vmax = max(m.max() for m in all_vals)

    for ag in range(n_groups):
        for h in range(ATT_HEADS):
            ax  = axes[ag][h]
            key = (stim, ag, h)
            if key not in att_dict:
                ax.axis("off")
                continue
            show_cb = (h == ATT_HEADS - 1)
            plot_heatmap(ax, att_dict[key],
                         title=f"{AGE_LABELS[ag]}\nHead {h+1}",
                         vmin=vmin, vmax=vmax, show_cbar=show_cb)

    plt.tight_layout()
    return fig


def make_summary_figure(att_dict):
    """Average over all stimuli → one summary figure."""
    n_groups = len(AGE_GROUPS)
    fig, axes = plt.subplots(n_groups, ATT_HEADS,
                             figsize=(5 * ATT_HEADS, 4.5 * n_groups),
                             squeeze=False)
    fig.suptitle("Cross-channel attention (averaged over all stimuli)", fontsize=12, fontweight="bold")

    # Accumulate grand-mean per (age_group, head)
    grand = {}
    counts = {}
    for (stim, ag, h), mat in att_dict.items():
        k = (ag, h)
        grand.setdefault(k, np.zeros((14, 14)))
        counts.setdefault(k, 0)
        grand[k] += mat
        counts[k] += 1
    grand = {k: grand[k] / counts[k] for k in grand}

    all_vals = list(grand.values())
    vmin = min(m.min() for m in all_vals)
    vmax = max(m.max() for m in all_vals)

    for ag in range(n_groups):
        for h in range(ATT_HEADS):
            ax = axes[ag][h]
            k  = (ag, h)
            if k not in grand:
                ax.axis("off")
                continue
            show_cb = (h == ATT_HEADS - 1)
            plot_heatmap(ax, grand[k],
                         title=f"{AGE_LABELS[ag]}\nHead {h+1}",
                         vmin=vmin, vmax=vmax, show_cbar=show_cb)

    plt.tight_layout()
    return fig, grand


def make_difference_figure(grand):
    """Young vs Old difference heatmap, per head."""
    fig, axes = plt.subplots(1, ATT_HEADS, figsize=(5 * ATT_HEADS, 4.5), squeeze=False)
    fig.suptitle("Attention difference: Older − Young (averaged over all stimuli)",
                 fontsize=12, fontweight="bold")

    diffs = []
    for h in range(ATT_HEADS):
        young_k = (0, h)   # age group 0 = young
        old_k   = (2, h)   # age group 2 = old
        if young_k in grand and old_k in grand:
            diffs.append(grand[old_k] - grand[young_k])
        else:
            diffs.append(None)

    vext = max(abs(d).max() for d in diffs if d is not None)

    for h, diff in enumerate(diffs):
        ax = axes[0][h]
        if diff is None:
            ax.axis("off")
            continue
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vext, vmax=vext, aspect="auto")
        ax.set_xticks(range(14)); ax.set_xticklabels(CH_LABELS, rotation=90, fontsize=7)
        ax.set_yticks(range(14)); ax.set_yticklabels(CH_LABELS, fontsize=7)
        ax.set_title(f"Head {h+1}", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ attention")

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    att_dict = collect_attention()

    stimuli = sorted({k[0] for k in att_dict})
    print(f"Found {len(stimuli)} stimuli, {len(att_dict)} (stim, age_group, head) combinations")

    # Per-stimulus figures
    stim_dir = os.path.join(OUT_DIR, "per_stimulus")
    os.makedirs(stim_dir, exist_ok=True)
    for stim in tqdm(stimuli, desc="Plotting stimuli"):
        fig = make_stimulus_figure(att_dict, stim)
        if fig is None:
            continue
        safe = stim.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(stim_dir, f"att_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary + difference figures
    fig_sum, grand = make_summary_figure(att_dict)
    fig_sum.savefig(os.path.join(OUT_DIR, "F6_attention_summary.png"), dpi=200, bbox_inches="tight")
    fig_sum.savefig(os.path.join(OUT_DIR, "F6_attention_summary.pdf"), bbox_inches="tight")
    plt.close(fig_sum)

    fig_diff = make_difference_figure(grand)
    fig_diff.savefig(os.path.join(OUT_DIR, "F6_attention_diff_old_minus_young.png"), dpi=200, bbox_inches="tight")
    fig_diff.savefig(os.path.join(OUT_DIR, "F6_attention_diff_old_minus_young.pdf"), bbox_inches="tight")
    plt.close(fig_diff)

    print(f"\nSaved:")
    print(f"  {OUT_DIR}/F6_attention_summary.png")
    print(f"  {OUT_DIR}/F6_attention_diff_old_minus_young.png")
    print(f"  {OUT_DIR}/per_stimulus/att_<stimulus>.png  ({len(stimuli)} files)")
