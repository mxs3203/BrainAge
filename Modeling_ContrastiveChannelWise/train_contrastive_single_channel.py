import json
import os

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from DatasetClasses.SupervisedContrastiveDataset import SupervisedContrastiveLearningDataset
from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder

from utils.embedding_metrics import alignment, uniformity, collapse_score
from utils.ContrastiveLoss import nt_xent_loss, supervised_contrastive_loss
from utils.Visualization import visualize_umap_age_and_id_structure

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]
target_align = 0.3
target_unif = -2.0

torch.backends.cudnn.benchmark = True

def train_channel(channel_idx, channel_name):
    os.makedirs("ContrastiveModel_chkpoints", exist_ok=True)

    with open("lets_try_simple_hyperparam_config.json", "r") as f:
        config = json.load(f)
    hyperparams = config[channel_name]
    LR = 1e-3
    hidden_size = hyperparams['hidden_size']
    num_layers_gru = hyperparams['num_layers_gru']
    projection_size = hyperparams['projection_size']
    cnn_dropout = hyperparams['CNN_dropout']
    rnn_dropout = hyperparams['RNN_dropout']
    activation = hyperparams['activation_function']
    rnn_type = hyperparams['rnn_type']
    num_epochs = 500
    TEMP = hyperparams['temp']
    BATCH_SIZE = 256
    AGE_THRS = 5

    kernel_size = hyperparams['kernel_size']
    stride = hyperparams['stride']
    padding = hyperparams['padding']
    channels = hyperparams['Conv1DChannels']

    dataset = SupervisedContrastiveLearningDataset(
        data_dir="../results/EEG_LSTM_windows/",
        channel_idx=channel_idx,
        subset_fraction=0.5,
        preload=False
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=10,pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=False, num_workers=10,pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model, device_ids=[0,1])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    wandb.init(
        project="BrainAge",
        entity="mxs3203",
        name="Contrastive_Channel_{}".format(channel_name),
        config={
            "model": "Contrastive_SingleChannel_{}".format(channel_name),
            "Channel": channel_name,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": LR,
            "FullData": True,
            "config_file": config
        }
    )

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0

        for i, (x1, x2, ages, meta) in enumerate(train_loader):
            x1, x2, ages = x1.to(device), x2.to(device), ages.to(device)
            optimizer.zero_grad()
            (z1,_), (z2,_) = model(x1), model(x2)
            loss = supervised_contrastive_loss(z1, z2,ages=ages, temperature=TEMP, mode="gaussian", gaussian_sigma=5.0)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)
        # Validation
        model.eval()
        total_val_loss = 0
        alignments, uniformities = [], []
        with torch.no_grad():
            for i, (x1, x2, ages, meta) in enumerate(val_loader):
                x1, x2, ages = x1.to(device), x2.to(device), ages.to(device)
                (z1,_), (z2,_) = model(x1), model(x2)
                loss = supervised_contrastive_loss(z1, z2,ages=ages, temperature=TEMP, mode="gaussian", gaussian_sigma=5.0)
                total_val_loss += loss.item()
                alignments.append(alignment(z1, z2).item())
                uniformities.append(uniformity(torch.cat([z1, z2], dim=0)).item())
            mean_std, dead_frac = collapse_score(torch.cat([z1, z2], dim=0))
            mean_std, dead_frac = mean_std.item(), dead_frac.item()

        if epoch >= 1 and epoch % 1 == 0:
            model.eval()
            all_z, all_ages, all_ids = [], [], []
            with torch.no_grad():
                for x1_u, x2_u, ages_u, meta_u in val_loader:
                    x1_u, x2_u = x1_u.to(device), x2_u.to(device)
                    (z1_u, _), (z2_u, _) = model(x1_u), model(x2_u)
                    all_z.append(torch.cat([z1_u, z2_u], dim=0).cpu())
                    all_ages.append(torch.cat([ages_u, ages_u], dim=0))
                    all_ids.extend(list(meta_u["ID"]) + list(meta_u["ID"]))
            fig = visualize_umap_age_and_id_structure(
                Z=torch.cat(all_z, dim=0),
                age=torch.cat(all_ages, dim=0),
                ids=all_ids,
                epoch=epoch,
                title_prefix=f"UMAP_{channel_name}"
            )
            wandb.log({"UMAP": wandb.Image(fig)})
            plt.close(fig)
        total_val_loss /= len(val_loader)
        mean_align = sum(alignments) / len(alignments)
        mean_unif = sum(uniformities) / len(uniformities)


        score = (
                total_val_loss
                + 0.5 * abs(mean_align - target_align)
                + 0.5 * abs(mean_unif - target_unif)
        )
        wandb.log({
            "Epoch": epoch,
            "Train/Loss": total_train_loss,
            "Val/Loss": total_val_loss,
            "Val/Align": mean_align,
            "Val/Unif": mean_unif,
            "Val/ZStd": mean_std,
            "Val/DeadFrac": dead_frac,
            "Val/Score": score,
            "Val/EmbStd_min": torch.cat([z1, z2], dim=0).std(dim=0).min().item()
        })
        if epoch >= 10 and epoch % 10 == 0:
            torch.save(model.state_dict(), "ContrastiveModel_chkpoints/EEGContrsative_forCH_{}_Epoch{}.pth".format(channel_name, epoch))

    wandb.finish()

def main():
    for idx, ch_name in enumerate(channel_order):
        if ch_name not in [] :
            print(f"\n=== Training channel {idx}: {ch_name} ===")
            train_channel(idx, ch_name)
        else:
            print("skipping {}".format(ch_name))

if __name__ == "__main__":
    main()