import json
import os

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder

from utils.embedding_metrics import alignment, uniformity, collapse_score
from ModelDefs.ChannelEncoderLSTM import ChannelEncoder
from DatasetClasses.DataDrivenContrastiveDataset import EEGContrastivePairs
from utils.UnsupervisedContrastiveLoss import nt_xent_loss
from utils.Visualization import visualize_umap

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

def collate_pairs(batch):
    X1 = torch.stack([b[0] for b in batch], dim=0)  # [B, T, C]
    X2 = torch.stack([b[1] for b in batch], dim=0)
    meta = {
        "ID": [b[2]["ID"] for b in batch],
        "stimulus": [b[2]["stimulus"] for b in batch],
        "Age": [b[2]["Age"] for b in batch],
    }
    return X1, X2, meta

torch.backends.cudnn.benchmark = True

def train_channel(channel_idx, channel_name):
    os.makedirs("ContrastiveModel_chkpoints", exist_ok=True)

    with open("../FindOptimalProbForAugmentation/contrastive_hyperparams.json", "r") as f:
        config = json.load(f)
    hyperparams = config[channel_name]
    LR = 1e-4
    hidden_size = hyperparams['hidden_size']
    num_layers_gru = hyperparams['num_layers_gru']
    projection_size = hyperparams['projection_size']
    num_epochs = 2000
    TEMP = hyperparams['temp']
    BATCH_SIZE = 256#hyperparams['batch_size']

    kernel_size = hyperparams['kernel_size']
    kernel_size2 = hyperparams['kernel_size2']
    stride2 = hyperparams['stride2']
    stride = hyperparams['stride']
    padding2 = hyperparams['padding2']
    padding = hyperparams['padding']
    channels = hyperparams['channels']

    dataset = EEGContrastivePairs(
        data_dir="../results/EEG_LSTM_windows/",
        channel_idx=channel_idx,
        subset_fraction=1.0
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=10,pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=False, num_workers=10,pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            "hidden_size": hidden_size,
            "input_size": 1 if channel_idx != "all" else len(channel_order),
            "num_layers_gru": num_layers_gru,
            "temperature": TEMP,
            "FullData": True
        }
    )

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
        # Validation
        model.eval()
        total_val_loss = 0
        alignments, uniformities = [], []
        with torch.no_grad():
            for i, (x1, x2, meta) in enumerate(val_loader):
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)
                total_val_loss += loss.item()
                alignments.append(alignment(z1, z2).item())
                uniformities.append(uniformity(torch.cat([z1, z2], dim=0)).item())
            Z = torch.cat([z1, z2], dim=0)
            if epoch >= 500 and epoch % 200 == 0:
                age = torch.cat([
                    torch.as_tensor(meta["Age1"], dtype=torch.float32),
                    torch.as_tensor(meta["Age2"], dtype=torch.float32)
                ], dim=0)
                fig = visualize_umap(Z, age, title=f"UMAP_{channel_name}_{epoch}")
                wandb.log({"UMAP": wandb.Image(fig)})
                plt.close(fig)
            mean_std, dead_frac = collapse_score(Z)
            mean_std, dead_frac = mean_std.item(), dead_frac.item()
        total_val_loss /= len(val_loader)
        mean_align = sum(alignments) / len(alignments)
        mean_unif = sum(uniformities) / len(uniformities)

        score = total_val_loss + 0.1 * mean_align + 0.1 * abs(mean_unif)
        wandb.log({
            "Epoch": epoch,
            "Train/Loss": total_train_loss,
            "Val/Loss": total_val_loss,
            "Val/Align": mean_align,
            "Val/Unif": mean_unif,
            "Val/ZStd": mean_std,
            "Val/DeadFrac": dead_frac,
            "Val/Score": score
        })
        if epoch >= 900 and epoch % 100 == 0:
            torch.save(model.state_dict(), "ContrastiveModel_chkpoints/EEGContrsative_forCH_{}_Epoch{}.pth".format(channel_name, epoch))

    wandb.finish()

def main():
    for idx, ch_name in enumerate(channel_order):
        if ch_name not in ['EEG.AF3'] :
            print(f"\n=== Training channel {idx}: {ch_name} ===")
            train_channel(idx, ch_name)
        else:
            print("skipping {}".format(ch_name))

if __name__ == "__main__":
    main()