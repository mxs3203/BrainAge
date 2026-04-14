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
    "Channel", "Trial", "HiddenSize", "GRULayers", "ProjSize", "Temp", "GaussianSigma","ActivationFunction",
    "RNN_dropout", "RNN_Type",
    "Train_Loss", "Val_Loss", "Alignment", "Uniformity",
    "Collapse_MeanStd", "Collapse_DeadFrac",
    "Score", "ModelType", "KernelSize", "CNN_dropout", "Padding", "Stride"
])
results_lock = threading.Lock()


def train_channel(trial, channel_name, dataset):
    LR = 1e-3
    num_epochs = 20
    TEMP = trial.suggest_categorical("temp", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    BATCH_SIZE = 512

    model_type = 'CONVGRU'

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            drop_last=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_size = trial.suggest_int("hidden_size", 2, 20)
    num_layers_gru = trial.suggest_int("num_layers_gru", 1, 2)
    rnn_dropout = trial.suggest_categorical("RNN_dropout", [0.01, 0.05,0.1,0.15, 0.2])
    cnn_dropout = trial.suggest_categorical("CNN_dropout", [0.01, 0.05,0.1,0.15, 0.2])
    projection_size = trial.suggest_int("projection_size", 10, 20)
    activation = trial.suggest_categorical("activation_function", ['ReLU', 'GELU', 'SiLU'])
    rnn_type = trial.suggest_categorical("rnn_type", ['GRU'])
    kernel_size = trial.suggest_int("kernel_size", 3, 9)
    stride = trial.suggest_int("stride", 1, 2)
    padding = trial.suggest_int("padding", 1, 4)
    channels = trial.suggest_int("Conv1DChannels", 2, 20)
    gaussian_sigma = trial.suggest_categorical("Gaussian_sigma", [2.0, 3.0, 4.0])

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

    target_align = 0.2
    target_unif  = -2.0

    def run_validation():
        model.eval()
        val_loss_total, alignments, uniformities = 0, [], []
        last_z1, last_z2 = None, None
        with torch.no_grad():
            for x, ages, meta in val_loader:
                x, ages = x.to(device), ages.to(device)
                with autocast(device_type="cuda"):
                    z1, _ = model(augmentor(x))
                    z2, _ = model(augmentor(x))
                loss = supervised_contrastive_loss(z1.float(), z2.float(), ages=ages, temperature=TEMP,
                                                   mode="gaussian", gaussian_sigma=gaussian_sigma)
                val_loss_total += loss.item()
                alignments.append(alignment(z1, z2).item())
                uniformities.append(uniformity(torch.cat([z1, z2], dim=0)).item())
                last_z1, last_z2 = z1, z2
        m_align = sum(alignments) / len(alignments)
        m_unif  = sum(uniformities) / len(uniformities)
        m_std, d_frac = collapse_score(torch.cat([last_z1, last_z2], dim=0))
        v_loss = val_loss_total / len(val_loader)
        sc = v_loss + 0.5 * max(0.0, m_align - target_align) + 0.5 * max(0.0, m_unif - target_unif)
        model.train()
        return v_loss, m_align, m_unif, m_std.item(), d_frac.item(), sc

    # Training loop
    total_train_loss = total_val_loss = mean_align = mean_unif = mean_std = dead_frac = score = 0
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0
        for x, ages, meta in train_loader:
            x, ages = x.to(device), ages.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                z1, _ = model(augmentor(x))
                z2, _ = model(augmentor(x))
            loss = supervised_contrastive_loss(z1.float(), z2.float(), ages=ages, temperature=TEMP,
                                               mode="gaussian", gaussian_sigma=gaussian_sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)

        # Validate at epoch 1 and every 5 epochs after
        if epoch == 1 or epoch % 5 == 0:
            total_val_loss, mean_align, mean_unif, mean_std, dead_frac, score = run_validation()
            print(f"Epoch {epoch}/{num_epochs} | Train: {total_train_loss:.4f} "
                  f"Val: {total_val_loss:.4f} Align: {mean_align:.4f} "
                  f"Unif: {mean_unif:.4f} Score: {score:.4f}")

            # Collapse detection at epoch 1 — prune immediately if degenerate
            if epoch == 1 and (mean_unif > -0.1 or mean_align < 0.001 or dead_frac > 0.5):
                del model, augmentor, optimizer, scaler
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            trial.report(score, epoch)
            if trial.should_prune():
                del model, augmentor, optimizer, scaler
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

    print(f"Final | Val: {total_val_loss:.4f} Align: {mean_align:.4f} Unif: {mean_unif:.4f} Score: {score:.4f}")

    trial_results = {
        "Channel": channel_name,
        "Trial": trial.number,
        "HiddenSize": hidden_size,
        "GRULayers": num_layers_gru,
        "ProjSize": projection_size,
        "Temp": TEMP,
        "GaussianSigma": gaussian_sigma,
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
        results.to_csv("contrastive_optuna_results{}.csv".format(channel_name), index=False)

    del model, augmentor, optimizer, scaler
    del train_loader, val_loader, train_set, val_set
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, nargs=2, default=None,
                        help="Start and end index of channels to optimise, e.g. --channels 0 7. "
                             "Omit to run all channels (single-GPU default mode).")
    args = parser.parse_args()

    if args.channels is not None:
        start, end = args.channels
        subset = list(enumerate(channel_order))[start:end]
    else:
        subset = list(enumerate(channel_order))

    for idx, ch_name in subset:
        if ch_name not in []:
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
            # split mode: each process owns one GPU -> fixed 5 parallel trials
            # default mode: scale with however many GPUs are visible
            n_jobs = 5 if args.channels is not None else (torch.cuda.device_count() * 5 if torch.cuda.is_available() else 1)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining = max(0, 120 - completed)
            print(f"Completed trials: {completed} | Remaining: {remaining}")
            if remaining > 0:
                study.optimize(
                    partial(train_channel, channel_name=ch_name, dataset=dataset),
                    n_trials=remaining,
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
