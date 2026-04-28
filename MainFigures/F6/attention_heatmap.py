"""
attention_heatmap.py
====================
Runs the full dataset through the AgeModel once, collects MultiheadAttention
weights [B, num_heads, 14, 14], and saves three cache files:
  attention_cache_3bin.pkl
  attention_cache_4bin.pkl
  attention_cache_5bin.pkl

Each cache: (stimulus, bin_idx, head_idx) -> mean [14,14] attention matrix.
The raw accumulator (keyed by exact age) is also saved as attention_cache_raw.pkl
so re-binning is instant without re-running the model.
"""
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
ATT_CACHE   = os.path.join(_SCRIPT_DIR, "attention_cache.pkl")          # legacy name (unused)
RAW_CACHE   = os.path.join(_SCRIPT_DIR, "attention_cache_raw.pkl")       # per exact-age accumulator
STIM_CSV    = os.path.join(_BRAINAGE, "stimuli_descriptions_and_expectations.csv")
OUT_DIR     = _SCRIPT_DIR

# Allowed stimuli: from CSV, excluding resting baselines
_BASELINES    = {"eyesopen", "eyesclose"}
_stim_df      = pd.read_csv(STIM_CSV)
ALLOWED_STIMS = set(_stim_df["original_stimuli_name"]) - _BASELINES
print(f"Allowed stimuli: {len(ALLOWED_STIMS)}  {sorted(ALLOWED_STIMS)}")

BATCH_SIZE  = 9000
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

# Age binning configurations — one cache will be produced per config
BIN_CONFIGS = {
    3: [(0, 30), (30, 50), (50, 200)],
    4: [(0, 27), (27, 34), (34, 45), (45, 200)],
    5: [(0, 25), (25, 31), (31, 39), (39, 47), (47, 200)],
}

# Raw accumulator uses exact integer age — re-aggregated into bins at save time
def age_to_int(age):
    return int(age)

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


def assign_bin(age, groups):
    for i, (lo, hi) in enumerate(groups):
        if lo <= age < hi:
            return i
    return len(groups) - 1


def aggregate_bins(raw_acc, groups):
    """Merge raw (stim, age_int, head) accumulator into (stim, bin_idx, head) means."""
    merged = {}
    for (stm, age_int, h), (mat_sum, cnt) in raw_acc.items():
        b = assign_bin(age_int, groups)
        key = (stm, b, h)
        if key not in merged:
            merged[key] = [np.zeros((14, 14)), 0]
        merged[key][0] += mat_sum
        merged[key][1] += cnt
    return {k: v[0] / v[1] for k, v in merged.items()}


# ── Collect raw cache (keyed by exact age) then fan out to bin caches ─────────
def collect_attention():
    """
    1. Run forward pass once; accumulate by (stim, age_int, head).
    2. Re-aggregate into 3-bin, 4-bin, 5-bin versions and save each.
    Returns the 5-bin cache (default).
    """
    if os.path.exists(RAW_CACHE):
        print(f"Loading raw cache from {RAW_CACHE} …")
        with open(RAW_CACHE, "rb") as f:
            raw_acc = pickle.load(f)
    else:
        print("No raw cache — running full dataset forward pass …")

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

        index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
        loader = DataLoader(
            EEGAttDataset(index_df),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=True, collate_fn=collate_fn,
        )

        # Accumulate by exact integer age — no binning yet
        raw_acc = {}   # (stim, age_int, head) -> [sum_matrix, count]

        with torch.no_grad():
            for X_batch, ages, stims in tqdm(loader, desc="Collecting attention"):
                if X_batch is None:
                    continue
                X_batch = X_batch.to(DEVICE)
                _, att = model(X_batch)
                att_np = att.cpu().float().numpy()   # [B, heads, 14, 14]

                for b in range(len(ages)):
                    stm = stims[b]
                    if stm not in ALLOWED_STIMS:
                        continue
                    age_int = age_to_int(ages[b])
                    for h in range(ATT_HEADS):
                        key = (stm, age_int, h)
                        mat = att_np[b, h]
                        if key not in raw_acc:
                            raw_acc[key] = [np.zeros((14, 14)), 0]
                        raw_acc[key][0] += mat
                        raw_acc[key][1] += 1

        with open(RAW_CACHE, "wb") as f:
            pickle.dump(raw_acc, f)
        print(f"Saved raw cache → {RAW_CACHE}")

    # Fan out to one cache per bin configuration
    for n_bins, groups in BIN_CONFIGS.items():
        out_path = os.path.join(_SCRIPT_DIR, f"attention_cache_{n_bins}bin.pkl")
        result   = aggregate_bins(raw_acc, groups)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        stims_found = len({k[0] for k in result})
        print(f"Saved {n_bins}-bin cache → {out_path}  "
              f"({len(result)} keys, {stims_found} stimuli)")

    return aggregate_bins(raw_acc, BIN_CONFIGS[5])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collect_attention()
    print("\nAll caches ready:")
    for n in (3, 4, 5):
        p = os.path.join(_SCRIPT_DIR, f"attention_cache_{n}bin.pkl")
        size = os.path.getsize(p) // 1024 if os.path.exists(p) else 0
        print(f"  attention_cache_{n}bin.pkl  ({size} KB)")
