import json
import os
import random
import sys

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sympy import true
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import wandb
from DatasetClasses.AgeWeightedDataset import EEGIndexedDataset
from ModelDefs.AttAgeModel import AgeModel
from Modeling_ContrastiveChannelWise.visualize_model import visualize_all_channels, visualize_integrated_gradients_all_channels
from utils.weighted_sampling_age import compute_sample_weights, split_index_by_participant, age_to_bin, \
    load_or_create_index

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

BATCH_SIZE = 256
LR = 1e-5
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "../results/EEG_LSTM_windows/"
PKL_PATH = "eeg_window_index.pkl"

index_df = load_or_create_index(ROOT_DIR, PKL_PATH)
index_df["age_bin"] = index_df["Age"].apply(age_to_bin)

train_index_df, val_index_df = split_index_by_participant(index_df)

train_weights = compute_sample_weights(train_index_df)
train_index_df["sample_weight"] = train_weights

train_dataset = EEGIndexedDataset(train_index_df, channels="all", target="Age", transform=None)
val_dataset = EEGIndexedDataset(val_index_df, channels="all", target="Age", transform=None)

train_index_df.to_csv("eeg_window_index_train.csv", index=False)
val_index_df.to_csv("eeg_window_index_val.csv", index=False)

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(train_index_df["sample_weight"].values),
    num_samples=len(train_index_df),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    pin_memory=True,
    num_workers=20,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=20,
    persistent_workers=True
)

model = AgeModel(attention_heads=2, unlock_encoders=True)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model, device_ids=[0, 1])

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criteria = nn.SmoothL1Loss(reduction='mean')

wandb.init(
    project="BrainAge",
    entity="mxs3203",
    config={
        "batch_size": train_loader.batch_size,
        "learning_rate": LR,
        "random_seed": random.Random(random.randrange(sys.maxsize))
    }
)
for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    total_train_loss = 0
    #dataset.refresh_subset()

    for i, (x, y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred,att = model(x)
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
            y_pred,att = model(x)
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
    if epoch % 2 == 0:
        torch.save(model.state_dict(), "ContrastiveModel_chkpoints/FullModel_Epoch{}.pth".format(epoch))

wandb.finish()
