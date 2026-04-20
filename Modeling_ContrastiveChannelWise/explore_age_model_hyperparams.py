import json
import torch
import optuna
from torch import nn, optim
from torch.amp import autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from DatasetClasses.AgeWeightedDataset import EEGIndexedDataset
from ModelDefs.AttAgeModel import AgeModel
from utils.weighted_sampling_age import (
    compute_sample_weights, split_index_by_participant,
    age_to_bin, load_or_create_index
)

with open("../config.json", "r") as f:
    config = json.load(f)

BATCH_SIZE       = 2048
NUM_EPOCHS       = 10
NUM_TRIALS       = 100
SEED             = 27
ROOT_DIR         = "../results/EEG_LSTM_windows/"
PKL_PATH         = "eeg_window_index.pkl"
STORAGE          = "sqlite:///age_model_optuna.db"
STUDY_NAME       = "AgeModel_hyperparam_search"
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Valid (common_dim, attention_heads) pairs — dim must be divisible by heads
DIM_HEAD_COMBOS = [
    "16_1", "16_2",
    "32_1", "32_2", "32_4",
    "64_1", "64_2", "64_4",
]

# ---------------------------------------------------------------------------
# Build data loaders once — reused across all trials
# ---------------------------------------------------------------------------
index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
index_df["age_bin"] = index_df["Age"].apply(age_to_bin)
train_index_df, val_index_df = split_index_by_participant(index_df, n_participants=30)
train_weights = compute_sample_weights(train_index_df)
train_index_df["sample_weight"] = train_weights

train_dataset = EEGIndexedDataset(train_index_df, channels="all", target="Age", transform=None)
val_dataset   = EEGIndexedDataset(val_index_df,   channels="all", target="Age", transform=None)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial):
    # Each trial gets its own DataLoaders — required for n_jobs > 1 (shared
    # loaders with persistent_workers are not safe across concurrent threads)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(train_index_df["sample_weight"].values),
        num_samples=len(train_index_df),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              pin_memory=True, num_workers=8, persistent_workers=True,
                              worker_init_fn=lambda wid: torch.manual_seed(SEED + wid))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=8, persistent_workers=True,
                              worker_init_fn=lambda wid: torch.manual_seed(SEED + wid))
    combo           = trial.suggest_categorical("dim_heads", DIM_HEAD_COMBOS)
    common_dim, attention_heads = map(int, combo.split("_"))
    fc1_dim         = trial.suggest_categorical("fc1_dim", [16, 32, 64, 128, 256, 512])
    fc2_dim         = trial.suggest_categorical("fc2_dim", [16, 32, 64, 128, 256, 512])
    unlock_encoders = trial.suggest_categorical("unlock_encoders", [False])
    dropout         = trial.suggest_float("dropout", 0.01, 0.3)
    lr              = 1e-6
    activation      = trial.suggest_categorical("activation", ["SiLU", "GELU", "ReLU"])

    model = AgeModel(
        attention_heads=attention_heads,
        common_dim=common_dim,
        dropout=dropout,
        fc1_dim=fc1_dim,
        fc2_dim=fc2_dim,
        activation=activation,
        unlock_encoders=unlock_encoders,
        hyperparam_config="contrastive_hyperparams.json"
    ).to(DEVICE)

    if unlock_encoders:
        optimizer = optim.Adam([
            {'params': model.encoders.parameters(),           'lr': lr * 0.01},
            {'params': model.channel_projectors.parameters(), 'lr': lr},
            {'params': model.attention.parameters(),          'lr': lr},
            {'params': model.fc1.parameters(),                'lr': lr},
            {'params': model.fc2.parameters(),                'lr': lr},
            {'params': model.output.parameters(),             'lr': lr},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    criteria = nn.SmoothL1Loss(reduction='mean')

    prev_val_loss = float("inf")
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        # --- train ---
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                y_pred, _ = model(x)
                loss = criteria(y, y_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred, _ = model(x)
                total_val_loss += criteria(y, y_pred).item()
        total_val_loss /= len(val_loader)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss

        # Prune if val loss went up compared to previous epoch
        if total_val_loss > prev_val_loss:
            del model, optimizer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        prev_val_loss = total_val_loss

        trial.report(total_val_loss, epoch)
        if trial.should_prune():
            del model, optimizer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    del model, optimizer
    torch.cuda.empty_cache()
    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study  = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, NUM_TRIALS - completed)
    print(f"Completed trials: {completed} / {NUM_TRIALS}. Running {remaining} more.")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=10)

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Val loss: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save best params to JSON
    out_path = "age_model_best_hyperparams.json"
    with open(out_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nBest hyperparams saved to {out_path}")


if __name__ == "__main__":
    main()
