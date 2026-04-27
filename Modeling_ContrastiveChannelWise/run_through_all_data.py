import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ModelDefs.AttAgeModel import AgeModel
from utils.weighted_sampling_age import load_or_create_index, age_to_bin

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT   = "ContrastiveModel_chkpoints/FullModel_Epoch_v4_46_bn_calibrated.pth"
ROOT_DIR     = "../results/EEG_LSTM_windows/"
PKL_PATH     = "eeg_window_index.pkl"
OUTPUT_CSV   = "results.csv"
BATCH_SIZE   = 8000
NUM_WORKERS  = 20

TRAIN_INDEX_CSV = "eeg_window_index_train.csv"
VAL_INDEX_CSV   = "eeg_window_index_val.csv"

# Model hyperparams (must match the checkpoint)
ATT_HEADS  = 2
COMMON_DIM = 16
FC1_DIM    = 512
FC2_DIM    = 512
ACTIVATION = "SiLU"
DROPOUT    = 0.1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
class EEGInferenceDataset(Dataset):
    """Returns (X, true_age, stimulus, participant_id) for every window."""
    def __init__(self, index_df):
        self.index_df = index_df.reset_index(drop=True)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        try:
            with open(row["file_path"], "rb") as fh:
                sample = pickle.load(fh)
            # Accept 'X' or 'x' as the data key
            data = sample.get("X", sample.get("x"))
            if data is None:
                raise KeyError(f"No 'X' key in {row['file_path']} — keys: {list(sample.keys())}")
            X        = torch.tensor(data, dtype=torch.float32)   # (512, 14)
            true_age = float(sample.get("Age", float("nan")))
            stimulus = str(sample.get("stimulus", ""))
            pid      = str(sample.get("ID", ""))
            return X, true_age, stimulus, pid
        except Exception as e:
            print(f"\n[SKIP] {row['file_path']}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, [], [], []
    Xs, ages, stimuli, pids = zip(*batch)
    return torch.stack(Xs), list(ages), list(stimuli), list(pids)


# ── Load index ───────────────────────────────────────────────────────────────
print("Loading index …")
index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
index_df["age_bin"] = index_df["Age"].apply(age_to_bin)

train_ids = pd.read_csv(TRAIN_INDEX_CSV)["ID"].unique() if os.path.exists(TRAIN_INDEX_CSV) else None
val_ids   = pd.read_csv(VAL_INDEX_CSV)["ID"].unique()   if os.path.exists(VAL_INDEX_CSV)   else None
print(f"  Total windows: {len(index_df):,}")

dataset    = EEGInferenceDataset(index_df)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Loading model from {CHECKPOINT} …")
model = AgeModel(
    attention_heads=ATT_HEADS,
    common_dim=COMMON_DIM,
    fc1_dim=FC1_DIM,
    fc2_dim=FC2_DIM,
    activation=ACTIVATION,
    dropout=DROPOUT,
    unlock_encoders=False,
    hyperparam_config="contrastive_hyperparams.json",
)

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

if torch.cuda.device_count() > 1:
    print(f"  Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

model = model.to(DEVICE)
model.eval()

# ── Inference ────────────────────────────────────────────────────────────────
records = []
with torch.no_grad():
    for X_batch, true_ages, stimuli, pids in tqdm(dataloader, desc="Predicting"):
        if X_batch is None:
            continue
        X_batch = X_batch.to(DEVICE)
        preds, _ = model(X_batch)
        preds = preds.cpu().numpy()

        for pred, age, stim, pid in zip(preds, true_ages, stimuli, pids):
            records.append({
                "ID":            pid,
                "stimulus":      stim,
                "true_age":      age,
                "predicted_age": float(pred),
            })

# ── Save ─────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df):,} predictions → {OUTPUT_CSV}")

def print_metrics(label, df):
    err = df["predicted_age"] - df["true_age"]
    huber = err.abs().apply(lambda e: 0.5 * e**2 if e < 1.0 else e - 0.5)
    n_participants = df["ID"].nunique()
    print(f"\n── {label} ({n_participants} participants, {len(df):,} windows) ──")
    print(f"  SmoothL1 (= WandB metric) = {huber.mean():.4f}")
    print(f"  MAE                       = {err.abs().mean():.3f} years")
    print(f"  RMSE                      = {((err**2).mean())**0.5:.3f} years")

print_metrics("ALL DATA", results_df)

if train_ids is not None:
    print_metrics("TRAIN participants", results_df[results_df["ID"].isin(train_ids)])

if val_ids is not None:
    print_metrics("VAL participants  ", results_df[results_df["ID"].isin(val_ids)])
