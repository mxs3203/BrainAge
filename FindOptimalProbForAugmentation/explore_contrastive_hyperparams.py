import json
from functools import partial

import optuna
import pandas as pd

import torch
from torch import nn, optim
from torch.nn.functional import dropout
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from utils.embedding_metrics import alignment, uniformity, collapse_score
from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder
from ModelDefs.ChannelEncoderLSTM import ChannelEncoder
from DatasetClasses.DataDrivenContrastiveDataset import EEGContrastivePairs
from utils.UnsupervisedContrastiveLoss import nt_xent_loss
with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

results = pd.DataFrame(columns=[
    "Channel", "Trial", "HiddenSize", "ProjSize", "Temp",
    "Train_Loss", "Val_Loss", "Alignment", "Uniformity",
    "Collapse_MeanStd", "Collapse_DeadFrac",
    "Score", "ModelType", "KernelSize", "Padding", "Stride",
    "KernelSize2", "Padding2", "Stride2"
])

def train_channel(trial,channel_idx, channel_name):
    LR = 1e-3

    num_epochs = 80
    TEMP = trial.suggest_float("temp", 0.05, 0.2)
    BATCH_SIZE = 256 #trial.suggest_int("batch_size", 256)

    model_type = 'CONVGRU'#trial.suggest_categorical(name='model', choices=['LSTM','CONVGRU'])

    dataset = EEGContrastivePairs(
        data_dir="../results/EEG_LSTM_windows/",
        channel_idx=channel_idx,
        subset_fraction=0.5
    )

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'LSTM':
        hidden_size = trial.suggest_int("hidden_size", 8, 64)
        num_lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        projection_size = trial.suggest_int("projection_size", 8, 16)
        dropout = trial.suggest_float("dropout", 0, 0.3)

        model = ChannelEncoder(
            input_size=1,
            lstm_hidden_size=hidden_size,
            lstm_layers=num_lstm_layers,
            proj_dim=projection_size,
            dropout=dropout
        )
    else:
        hidden_size = trial.suggest_int("hidden_size", 8, 64)
        num_layers_gru = trial.suggest_int("num_layers_gru", 1, 3)
        projection_size = trial.suggest_int("projection_size", 8, 16)
        kernel_size = trial.suggest_int("kernel_size", 2, 9)
        stride = trial.suggest_int("stride", 1, 2)
        padding = trial.suggest_int("padding", 1, 4)
        kernel_size2 = trial.suggest_int("kernel_size2", 2, 9)
        stride2 = trial.suggest_int("stride2", 1, 2)
        padding2 = trial.suggest_int("padding2", 1, 4)
        channels = trial.suggest_int("Conv1DChannels", 8, 128)
        model = EEGConvGRUEncoder(
            hidden_size=hidden_size,
            gru_hidden=hidden_size,
            embedding_dim=projection_size,
            num_layers_gru=num_layers_gru,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            kernel_size2=kernel_size2,
            stride2=stride2,
            padding2=padding2,
            conv_channels=channels
        )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model, device_ids=[0,1])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    # wandb.init(
    #     project="BrainAge",
    #     entity="mxs3203",
    #     config={
    #         "model": "Contrastive_SingleChannel{}".format(channel_name),
    #         "Channel": channel_name,
    #         "epochs": num_epochs,
    #         "batch_size": BATCH_SIZE,
    #         "learning_rate": LR,
    #         "hidden_size": hidden_size,
    #         "input_size": 1 if channel_idx != "all" else len(channel_order),
    #         "temperature": TEMP,
    #         "projection_size": projection_size,
    #         "modelType": model_type
    #     }
    # )

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0

        for i, (x1, x2, meta) in enumerate(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2, temperature=TEMP)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)
        print(
        f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {total_train_loss:.4f}")
        # wandb.log({
        #     "Train/Loss": total_train_loss,
        #     "Epoch": epoch
        # })

    # Validation
    model.eval()
    total_val_loss = 0
    alignments, uniformities = [], []
    with torch.no_grad():
        for i, (x1, x2, meta) in enumerate(val_loader):
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
    trial.report(score, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    print(
        f"Val Loss: {total_val_loss:.4f} Mean align: {mean_align} Mean Uniformity: {mean_unif} Total score: {score}")
    # wandb.log({
    #     "Val/Loss": total_val_loss ,
    #     "Val/Align" : mean_align,
    #     "Val/Unif": mean_unif,
    #     "Val/ZStd": mean_std,
    #     "Val/DeadFrac": dead_frac,
    #     "Val/Score": score
    # })
   # wandb.finish()

    trial_results = {
        "Channel": channel_name,
        "Trial": trial.number,
        "HiddenSize": hidden_size,
        "ProjSize": projection_size,
        "Temp": TEMP,
        "KernelSize": kernel_size,
        "Padding": padding,
        "Stride": stride,
        "KernelSize2": kernel_size2,
        "Padding2": padding2,
        "Stride2": stride2,
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
    results = pd.concat([results, pd.DataFrame([trial_results])], ignore_index=True)
    results.to_csv("contrastive_optuna_results.csv", index=False)


    return score

def main():
    for idx, ch_name in enumerate(channel_order):
        if ch_name not in [] :
            print(f"\n=== Training channel {idx}: {ch_name} ===")
            study = optuna.create_study(
                study_name=ch_name,
                storage="sqlite:///db.sqlite3",
                direction="minimize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=27)
            )
            study.optimize(partial(train_channel, channel_idx=idx, channel_name=ch_name), n_trials=100, n_jobs=1)

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