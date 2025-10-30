import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from Modeling_ContrastiveChannelWise.ChannelEncoder import ChannelEncoder
from Modeling_ContrastiveChannelWise.EEGAugmentor import EEGAugmentor
from Modeling_ContrastiveChannelWise.EEGContrastiveDataset import EEGDataset
from utils.UnsupervisedContrastiveLoss import nt_xent_loss

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]




def train_channel(channel_idx, channel_name):
    os.makedirs("ContrastiveModel_chkpoints", exist_ok=True)


    LR = 1e-3
    hidden_size = 32
    num_lstm_layers = 1
    projection_size = 16
    num_epochs = 50 + 1
    TEMP = 0.05
    BATCH_SIZE = 15000
    prob_jitter = 0.2
    prob_scale = 0.2
    prob_time_mask = 0.2
    augmentor = EEGAugmentor()

    def contrastive_transform(x):
        return augmentor(x.clone(), prob_jitter, prob_scale, prob_time_mask), augmentor(x.clone(), prob_jitter,
                                                                                        prob_scale, prob_time_mask)
    dataset = EEGDataset(root_dir="../results/EEG_LSTM_windows", channels=channel_idx, transform=contrastive_transform)


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
        name="Contrastive_Channel_{}".format(channel_name),
        config={
            "model": "Contrastive_SingleChannel{}".format(channel_name),
            "Channel": channel_name,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": LR,
            "hidden_size": hidden_size,
            "input_size": 1 if channel_idx != "all" else len(channel_order),
            "num_lstm_layers": num_lstm_layers,
            "temperature": TEMP
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

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x1, x2 in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)
                total_val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(val_loader):.4f}")
        wandb.log({
            "Train Loss": total_train_loss/len(train_loader),
            "Validation Loss": total_val_loss/len(val_loader),
            "Epoch": epoch + 1
        })
        if epoch % 5 == 0:
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