import json
import os
from functools import partial

import optuna
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from FindOptimalProbForAugmentation.embedding_metrics import alignment, uniformity, collapse_score
from Modeling_ContrastiveChannelWise.ChannelEncoder import ChannelEncoder
from Modeling_ContrastiveChannelWise.EEGAugmentor import EEGAugmentor
from Modeling_ContrastiveChannelWise.EEGContrastiveDataset import EEGDataset
from utils.UnsupervisedContrastiveLoss import nt_xent_loss
with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

results = pd.DataFrame(columns=[
    "Channel", "Trial", "HiddenSize", "ProjSize", "Temp",
    "Prob_Jitter", "Prob_Scale", "Prob_TimeMask",
    "Train_Loss", "Val_Loss", "Alignment", "Uniformity",
    "Collapse_MeanStd", "Collapse_DeadFrac",
    "Score"
])

def train_channel(trial,channel_idx, channel_name):
    LR = 1e-3
    hidden_size = trial.suggest_int("hidden_size", 8, 64)
    num_lstm_layers = 1
    projection_size = trial.suggest_int("projection_size", 8, 16)
    num_epochs = 50
    TEMP = trial.suggest_float("temp", 0.05, 0.15)
    BATCH_SIZE = 10000

    prob_jitter = trial.suggest_float("prob_jitter", 0.1, 0.5)
    prob_scale = trial.suggest_float("prob_scale", 0.1, 0.5)
    prob_time_mask = trial.suggest_float("prob_time_mask", 0.1, 0.3)
    augmentor = EEGAugmentor()

    def contrastive_transform(x):
        return augmentor(x.clone(), prob_jitter, prob_scale, prob_time_mask), augmentor(x.clone(), prob_jitter, prob_scale, prob_time_mask)
    dataset = EEGDataset(root_dir="../results/EEG_LSTM_windows", channels=channel_idx, transform=contrastive_transform, preload=True, subset_size=int(1e5))


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChannelEncoder(
        lstm_hidden_size=hidden_size,
        lstm_layers=num_lstm_layers,
        proj_dim=projection_size
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    wandb.init(
        project="BrainAge",
        entity="mxs3203",
        config={
            "model": "Contrastive_SingleChannel{}".format(channel_name),
            "Channel": channel_name,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": LR,
            "hidden_size": hidden_size,
            "input_size": 1 if channel_idx != "all" else len(channel_order),
            "num_lstm_layers": num_lstm_layers,
            "temperature": TEMP,
            "projection_size": projection_size,
            "probabilies" : [prob_jitter, prob_scale, prob_time_mask]
        }
    )

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0

        for x1, x2 in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2, temperature=TEMP)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print(
        f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {total_train_loss / len(train_loader):.4f}")
        wandb.log({
            "Train/Loss": total_train_loss / len(train_loader),
            "Epoch": epoch
        })
    # Validation
    model.eval()
    total_val_loss = 0
    alignments, uniformities = [], []
    with torch.no_grad():
        for x1, x2 in val_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2, temperature=TEMP)
            total_val_loss += loss.item()
            alignments.append(alignment(z1, z2).item())
            uniformities.append(uniformity(torch.cat([z1, z2], dim=0)).item())
        Z = torch.cat([z1, z2], dim=0)
        mean_std, dead_frac = collapse_score(Z)
        mean_std, dead_frac = mean_std.item(), dead_frac.item()
    total_val_loss /= len(val_loader)
    mean_align = sum(alignments) / len(alignments)
    mean_unif = sum(uniformities) / len(uniformities)

    score = total_val_loss + 0.1 * mean_align + 0.1 * abs(mean_unif)
    print(
        f"Val Loss: {total_val_loss:.4f} Mean align: {mean_align} Mean Uniformity: {mean_unif} Total score: {score}")
    wandb.log({
        "Val/Loss": total_val_loss ,
        "Val/Align" : mean_align,
        "Val/Unif": mean_unif,
        "Val/ZStd": mean_std,
        "Val/DeadFrac": dead_frac,
        "Val/Score": score
    })
    wandb.finish()

    trial_results = {
        "Channel": channel_name,
        "Trial": trial.number,
        "HiddenSize": hidden_size,
        "ProjSize": projection_size,
        "Temp": TEMP,
        "Prob_Jitter": prob_jitter,
        "Prob_Scale": prob_scale,
        "Prob_TimeMask": prob_time_mask,
        "Train_Loss": total_train_loss / len(train_loader),
        "Val_Loss": total_val_loss,
        "Alignment": mean_align,
        "Uniformity": mean_unif,
        "Collapse_MeanStd": mean_std,
        "Collapse_DeadFrac": dead_frac,
        "Score": score,
    }
    global results
    results = pd.concat([results, pd.DataFrame([trial_results])], ignore_index=True)
    results.to_csv("contrastive_optuna_results.csv", index=False)

    trial.report(score, step=0)
    return score

def main():
    for idx, ch_name in enumerate(channel_order):
        if ch_name not in ['EEG.O1','EEG.P7', 'EEG.FC5', 'EEG.T7', 'EEG.F3', 'EEG.F7', 'EEG.AF3', 'EEG.O2','EEG.P8'] :
            print(f"\n=== Training channel {idx}: {ch_name} ===")
            study = optuna.create_study(
                study_name=ch_name,
                storage="sqlite:///db.sqlite3",
                direction="minimize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(partial(train_channel, channel_idx=idx, channel_name=ch_name), n_trials=50, n_jobs=1)

            print("Best params:", study.best_params)
            print("Best score:", study.best_value)
        else:
            print("skipping {}".format(ch_name))

if __name__ == "__main__":
    main()

'''

Situation	        Alignment	        Uniformity	        Interpretation
✅ Healthy	        0.2 – 0.5	        −1.5 – −2.5	        Positives close, embeddings diverse.
⚠️ Collapsed(points same)≈ 0	        ≈ 0	                Model failed; loss near 0 but no structure.
⚠️ Over-spread	    > 1	                ≪ −3	            Embeddings uniform but positives far → under-invariance.
⚠️ Over-invariant	< 0.1	            > −1	            All positives overlap → lack diversity.

'''