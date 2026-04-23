import json
import os
import random
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.amp import GradScaler, autocast
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

#####################################

BATCH_SIZE = 128
LR = 0.6e-6
num_epochs = 100
att_heads = 2
common_dim = 16
fc1_dim = 512
fc2_dim = 512
activation = "SiLU"
UNLOCK_ENCODERS = True
DROPOUT = 0.1
AUTCAST = True
GRAD_SCALE = False
#####################################

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



model = AgeModel(attention_heads=att_heads, common_dim=common_dim,
                 fc1_dim=fc1_dim, fc2_dim=fc2_dim, activation=activation,
                 unlock_encoders=UNLOCK_ENCODERS,
                 dropout=DROPOUT,
                 hyperparam_config="contrastive_hyperparams.json")
path = os.path.join("ContrastiveModel_chkpoints/FullModel_Epoch0.pth")
print(path)
if os.path.exists(path):
    print(f"Continuing training from ... {path}")
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    state_dict = checkpoint

    # If model was trained with DataParallel, the keys will have 'module.' prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        print(f"Detected 'module.' prefix in the state_dict. Removing it...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
else:
    print("Missing Trained model at {path}")
    #raise FileNotFoundError(f"Missing Trained model at {path}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model, device_ids=[0, 1])

model = model.to(device)
base_model = model.module if isinstance(model, nn.DataParallel) else model
if base_model.unlock_encoders:
    optimizer = optim.Adam([
        {'params': base_model.encoders.parameters(),          'lr': LR * 0.01},
        {'params': base_model.channel_projectors.parameters(),'lr': LR},
        {'params': base_model.attention.parameters(),         'lr': LR},
        {'params': base_model.fc1.parameters(),               'lr': LR},
        {'params': base_model.fc2.parameters(),               'lr': LR},
        {'params': base_model.output.parameters(),            'lr': LR},
    ])
else:
    optimizer = optim.Adam(model.parameters(), lr=LR)
criteria = nn.SmoothL1Loss(reduction='mean')
scaler = GradScaler(device="cuda")
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)#, T_max=100, eta_min=1e-8)

wandb.init(
    project="BrainAge",
    entity="mxs3203",
    config={
        "batch_size": train_loader.batch_size,
        "learning_rate": LR,
        "att_heads": att_heads,
        "CommonDim": common_dim,
        "fc1_dim": fc1_dim,
        "fc2_dim": fc2_dim,
        "activation": activation,
        "dropout": DROPOUT,
        "unlock_encoders": UNLOCK_ENCODERS,
        "batch_size_per_gpu": BATCH_SIZE,
        "GradScale": GRAD_SCALE
    }
)
for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    total_train_loss = 0
    #dataset.refresh_subset()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            y_pred, att = model(x)
            loss = criteria(y, y_pred)
        if GRAD_SCALE:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    wandb.log({
        "Epoch": epoch,
        "Train/Loss": total_train_loss,
        "Val/Loss": total_val_loss,
        "LR": current_lr,
        "Conv Activations": wandb.Image(fig),
        "Integrated Gradients": wandb.Image(fig2)
    })
    plt.close(fig)
    plt.close(fig2)
    if epoch % 1 == 0:
        torch.save(model.state_dict(), "ContrastiveModel_chkpoints/FullModel_Epoch_v4_{}.pth".format(epoch))

wandb.finish()
