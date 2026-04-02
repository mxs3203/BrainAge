import json
import threading
from functools import partial

import optuna
import pandas as pd

import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from DatasetClasses.SupervisedContrastiveDataset import SupervisedContrastiveLearningDataset
from utils.embedding_metrics import alignment, uniformity, collapse_score
from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder
from utils.ContrastiveLoss import supervised_contrastive_loss
from utils.GpuAugmentor import EEGAugmentorGPU

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

results = pd.DataFrame(columns=[
    "Channel", "Trial", "HiddenSize", "GRULayers", "ProjSize", "Temp", "ActivationFunction",
    "RNN_dropout", "RNN_Type",
    "Train_Loss", "Val_Loss", "Alignment", "Uniformity",
    "Collapse_MeanStd", "Collapse_DeadFrac",
    "Score", "ModelType", "KernelSize", "CNN_dropout", "Padding", "Stride"
])
results_lock = threading.Lock()


def train_channel(trial, channel_idx, channel_name, dataset):
    LR = 1e-3
    num_epochs = 10
    TEMP = trial.suggest_float("temp", 0.05, 0.2)
    BATCH_SIZE = 256

    model_type = 'CONVGRU'

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            drop_last=False, num_workers=0)

    gpu_id = trial.number % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    hidden_size = trial.suggest_int("hidden_size", 2, 20)
    num_layers_gru = trial.suggest_int("num_layers_gru", 1, 2)
    rnn_dropout = trial.suggest_float("RNN_dropout", 0.05, 0.2)
    cnn_dropout = trial.suggest_float("CNN_dropout", 0.05, 0.2)
    projection_size = trial.suggest_int("projection_size", 10, 20)
    activation = trial.suggest_categorical("activation_function", ['ReLU', 'GELU', 'SiLU'])
    rnn_type = trial.suggest_categorical("rnn_type", ['GRU'])
    kernel_size = trial.suggest_int("kernel_size", 3, 9)
    stride = trial.suggest_int("stride", 1, 2)
    padding = trial.suggest_int("padding", 1, 4)
    channels = trial.suggest_int("Conv1DChannels", 2, 20)

    model = EEGConvGRUEncoder(
        gru_hidden=hidden_size,
        embedding_dim=projection_size,
        num_layers_gru=num_layers_gru,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        cnn_dropout=cnn_dropout,
        activation=activation,
        conv_channels=channels,
        rnn_dropout=rnn_dropout,
        rnn_type=rnn_type
    )
    model = model.to(device)
    augmentor = EEGAugmentorGPU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda")

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0

        for x, ages, meta in train_loader:
            x, ages = x.to(device), ages.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                z1, _ = model(augmentor(x))
                z2, _ = model(augmentor(x))
                loss = supervised_contrastive_loss(z1, z2, ages=ages, temperature=TEMP,
                                                   mode="gaussian", gaussian_sigma=5.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {total_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    alignments, uniformities = [], []
    with torch.no_grad():
        for x, ages, meta in val_loader:
            x, ages = x.to(device), ages.to(device)
            with autocast(device_type="cuda"):
                z1, _ = model(augmentor(x))
                z2, _ = model(augmentor(x))
                loss = supervised_contrastive_loss(z1, z2, ages=ages, temperature=TEMP,
                                                   mode="gaussian", gaussian_sigma=5.0)
            total_val_loss += loss.item()
            alignments.append(alignment(z1, z2).item())
            uniformities.append(uniformity(torch.cat([z1, z2], dim=0)).item())
        mean_std, dead_frac = collapse_score(torch.cat([z1, z2], dim=0))
        mean_std, dead_frac = mean_std.item(), dead_frac.item()
    total_val_loss /= len(val_loader)
    mean_align = sum(alignments) / len(alignments)
    mean_unif = sum(uniformities) / len(uniformities)

    target_align = 0.3
    target_unif = -2.0
    score = (
        total_val_loss
        + 0.5 * max(0.0, mean_align - target_align)
        + 0.5 * max(0.0, mean_unif - target_unif)
    )

    trial.report(score, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    print(f"Val Loss: {total_val_loss:.4f} Align: {mean_align:.4f} Unif: {mean_unif:.4f} Score: {score:.4f}")

    trial_results = {
        "Channel": channel_name,
        "Trial": trial.number,
        "HiddenSize": hidden_size,
        "GRULayers": num_layers_gru,
        "ProjSize": projection_size,
        "Temp": TEMP,
        "ActivationFunction": activation,
        "RNN_dropout": rnn_dropout,
        "RNN_Type": rnn_type,
        "KernelSize": kernel_size,
        "CNN_dropout": cnn_dropout,
        "Padding": padding,
        "Stride": stride,
        "Train_Loss": total_train_loss,
        "Val_Loss": total_val_loss,
        "Alignment": mean_align,
        "Uniformity": mean_unif,
        "Collapse_MeanStd": mean_std,
        "Collapse_DeadFrac": dead_frac,
        "Score": score,
        "ModelType": model_type
    }
    global results
    with results_lock:
        results = pd.concat([results, pd.DataFrame([trial_results])], ignore_index=True)
        results.to_csv("contrastive_optuna_results.csv", index=False)

    return score


def main():
    for idx, ch_name in enumerate(channel_order):
        if ch_name not in ['EEG.AF3', 'EEG.F7']:
            print(f"\n=== Training channel {idx}: {ch_name} ===")

            dataset = SupervisedContrastiveLearningDataset(
                data_dir="../results/EEG_LSTM_windows/",
                channel_idx=idx,
                subset_fraction=0.1,
                preload=True
            )

            study = optuna.create_study(
                study_name=ch_name,
                storage="sqlite:///db.sqlite3",
                direction="minimize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(n_startup_trials=20)
            )
            n_jobs = torch.cuda.device_count() * 10 if torch.cuda.is_available() else 1
            study.optimize(
                partial(train_channel, channel_idx=idx, channel_name=ch_name, dataset=dataset),
                n_trials=100,
                n_jobs=n_jobs
            )

            print("Best params:", study.best_params)
            print("Best score:", study.best_value)
        else:
            print("skipping {}".format(ch_name))


if __name__ == "__main__":
    main()

'''
Situation	        Alignment	        Uniformity	        Interpretation
✅ Healthy	        0.2 – 0.5	        −1.5 – −2.5	        Positives close, embeddings diverse.
⚠️ Collapsed        ≈ 0	                ≈ 0	                Model failed; loss near 0 but no structure.
⚠️ Over-spread	    > 1	                ≪ −3	            Embeddings uniform but positives far → under-invariance.
⚠️ Over-invariant	< 0.1	            > −1	            All positives overlap → lack diversity.
'''
