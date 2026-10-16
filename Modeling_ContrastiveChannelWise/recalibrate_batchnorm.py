"""
Recalibrate BatchNorm running statistics for a saved AgeModel checkpoint.

The problem: BatchNorm running_mean/running_var were accumulated during training
with WeightedRandomSampler, which oversamples certain age bins. At eval time,
these biased running stats cause wrong normalization → MAE jumps from ~6 to ~14.

The fix: reset running stats and recompute them on the UNWEIGHTED full dataset
by running a forward pass in train mode (which updates running stats) with
torch.no_grad() (so no gradients are computed or stored).
"""
import os
import pickle
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Always run from within Modeling_ContrastiveChannelWise/ regardless of how invoked
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from ModelDefs.AttAgeModel import AgeModel
from utils.weighted_sampling_age import load_or_create_index, age_to_bin


# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "ContrastiveModel_chkpoints/FullModel_Epoch_v4_9.pth"
OUTPUT_PATH  = "ContrastiveModel_chkpoints/FullModel_Epoch_v4_9_bn_calibrated.pth"
ROOT_DIR     = "../results/EEG_LSTM_windows/"
PKL_PATH     = "eeg_window_index.pkl"
BATCH_SIZE      = 512
NUM_WORKERS     = 8
MAX_CALIB_BATCHES = 500   # ~256k windows is more than enough for BN convergence

ATT_HEADS  = 2
COMMON_DIM = 16
FC1_DIM    = 512
FC2_DIM    = 512
ACTIVATION = "SiLU"
DROPOUT    = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGCalibDataset(Dataset):
    def __init__(self, index_df):
        self.index_df = index_df.reset_index(drop=True)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        try:
            with open(row["file_path"], "rb") as fh:
                sample = pickle.load(fh)
            data = sample.get("X", sample.get("x"))
            if data is None:
                return None
            return torch.tensor(data, dtype=torch.float32)
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack(batch)


def reset_batchnorm_stats(model):
    """Zero out running stats and batch counter so they're recomputed from scratch."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()


def calibrate(checkpoint_path, output_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = AgeModel(
        attention_heads=ATT_HEADS, common_dim=COMMON_DIM,
        fc1_dim=FC1_DIM, fc2_dim=FC2_DIM, activation=ACTIVATION,
        dropout=DROPOUT, unlock_encoders=False,
        hyperparam_config="contrastive_hyperparams.json",
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    print("Resetting BatchNorm running statistics …")
    reset_batchnorm_stats(model)

    print("Loading index …")
    index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
    dataset  = EEGCalibDataset(index_df)
    loader   = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn,
    )

    # Force ALL BatchNorm layers (including encoders) into train mode so running
    # stats get updated. AgeModel.train() keeps encoders in eval when unlock_encoders=False,
    # so we bypass that by setting each BN module directly.
    model.eval()  # start from eval so non-BN layers behave correctly
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()  # only BN layers in train mode → they update running stats
    n_calib = min(MAX_CALIB_BATCHES, len(loader))
    print(f"Running calibration pass: {n_calib} batches × {BATCH_SIZE} = ~{n_calib*BATCH_SIZE:,} windows …")
    with torch.no_grad():
        for i, X in enumerate(tqdm(loader, desc="BN calibration", total=n_calib)):
            if i >= MAX_CALIB_BATCHES:
                break
            if X is None:
                continue
            X = X.to(DEVICE)
            model(X)

    # Verify: switch to eval and run quick sanity check
    model.eval()
    print("\nSanity check (eval mode, first batch) …")
    with torch.no_grad():
        for X in loader:
            if X is None:
                continue
            X = X.to(DEVICE)
            preds, _ = model(X[:16])
            print(f"  Sample predictions (first 8): {preds[:8].squeeze().tolist()}")
            break

    torch.save(model.state_dict(), output_path)
    print(f"\nCalibrated checkpoint saved → {output_path}")


if __name__ == "__main__":
    calibrate(CHECKPOINT, OUTPUT_PATH)
