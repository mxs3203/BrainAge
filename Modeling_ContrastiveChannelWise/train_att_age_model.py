import json
import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sympy import true
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from DatasetClasses.EEGDataset import EEGDataset
from ModelDefs.AttAgeModel import AgeModel
from Modeling_ContrastiveChannelWise.visualize_model import visualize_all_channels, visualize_integrated_gradients_all_channels

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

def main():
    BATCH_SIZE = 512
    LR = 1e-4
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset(
        root_dir="../results/EEG_LSTM_windows/", channels="all", target="Age", transform=None, preload=False, subset_size=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=10, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = AgeModel(attention_heads=1, unlock_encoders=True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model, device_ids=[0, 1])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criteria = nn.MSELoss(reduction='mean')

    wandb.init(
        project="BrainAge",
        entity="mxs3203",
        config={
            "batch_size": train_loader.batch_size,
            "learning_rate": LR,
        }
    )
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_train_loss = 0
        dataset.refresh_subset()

        for i, (x, y) in tqdm(enumerate(train_loader)):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred,att, seg_lens = model(x)
            loss = criteria(y,y_pred )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        total_train_loss /= len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x,y = x.to(device), y.to(device)
                y_pred,att,seg_lens = model(x)
                loss = criteria(y, y_pred)
                total_val_loss += loss.item()
        total_val_loss /= len(val_loader)
        sample_idx = np.random.randint(0, np.shape(x)[0])  # random window
        fig = visualize_all_channels(model,x, channel_order, sample_idx=sample_idx)
        fig2 = visualize_integrated_gradients_all_channels(model, x, channel_order, sample_idx=sample_idx)
        wandb.log({
            "Epoch": epoch,
            "Train/Loss": total_train_loss,
            "Val/Loss": total_val_loss,
            "Conv Activations": wandb.Image(fig),
            "Integrated Gradients": wandb.Image(fig2)
        })
        plt.close(fig)
        plt.close(fig2)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "ContrastiveModel_chkpoints/FullModel_Epoch{}.pth".format(epoch))

    wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('fork', force=True)
    main()