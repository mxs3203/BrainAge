import json
import os

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from DatasetClasses.EEGDataset import EEGDataset
from ModelDefs.SingleCH_LSTM_Contrastive_Att_model import AgeModel

with open("../config.json", "r") as f:
    config = json.load(f)
channel_order = config["EEG_columns"]

def get_conv_activations(model, x):
    activations = {}
    def hook_fn(module, input, output):
        activations['conv_out'] = output.detach().cpu()
    h = model.conv.register_forward_hook(hook_fn)
    _ = model(x)
    h.remove()
    return activations['conv_out']

def plot_conv_activations(conv_act, num_filters=2):
    act = conv_act.squeeze(0)  # [num_filters, L']
    fig, axes = plt.subplots(num_filters, 1, figsize=(10, 6), sharex=True)
    for i in range(num_filters):
        axes[i].plot(act[i].numpy())
        axes[i].set_ylabel(f"F{i}")
    plt.suptitle("Conv1D Filter Activations")
    plt.savefig("convs.png")
    plt.show()

def plot_attention_matrix(attn_weights, eeg_channels, sample_idx=0, save_path="examlpe.png"):
    """
    attn_weights: [B, 14, 14] tensor
    eeg_channels: list of 14 channel names (order same as model.eeg_columns)
    """
    attn = attn_weights[sample_idx].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        attn
    )
    ax.set_title(f"Channel Attention Map – Sample {sample_idx}")
    ax.set_xlabel("Source (attended channel)")
    ax.set_ylabel("Target (query channel)")
    #plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_conv_activations_heatmap(conv_act, time_axis=None, max_filters=16, figsize=(10, 6)):
    """
    conv_act: Tensor of shape [1, num_filters, L'] or [1, num_filters, L', extra] or [num_filters, L']
    """
    act = conv_act.squeeze(0).detach().cpu()
    print("Activation shape:", act.shape)

    # If 3D -> collapse last dimension
    if act.ndim == 3:
        act = act.mean(dim=-1)  # average over extra dimension
    elif act.ndim == 1:
        act = act.unsqueeze(0)

    num_filters, L = act.shape
    if time_axis is None:
        time_axis = range(L)
    if num_filters > max_filters:
        act = act[:max_filters, :]

    plt.figure(figsize=figsize)
    sns.heatmap(
        act.numpy(),
        cmap="viridis",
        cbar_kws={"label": "Activation strength"},
        yticklabels=[f"F{i}" for i in range(act.shape[0])],
        xticklabels=round(len(time_axis) / 8)
    )
    plt.xlabel("Time (samples)")
    plt.ylabel("Convolution filters")
    plt.title("Conv1D Filter Activations as Heatmap")
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()

def main():
    BATCH_SIZE = 1024
    LR = 1e-3
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset(
        root_dir="../results/EEG_LSTM_windows/", channels="all", target="Age", transform=None, preload=False, subset_size=500000
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=10,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=10, pin_memory=True,
                            persistent_workers=True)

    model = AgeModel(
        attention_heads=2
    )

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
        plot_attention_matrix(att.unsqueeze(0), channel_order)
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        act = get_conv_activations(base_model.encoders[3], x[:, :, 3].unsqueeze(1))
        plot_conv_activations(act)
        plot_conv_activations_heatmap(act)
        wandb.log({
            "Epoch": epoch,
            "Train/Loss": total_train_loss,
            "Val/Loss": total_val_loss
        })
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), "ContrastiveModel_chkpoints/EEGContrsative_forCH_{}_Epoch{}.pth".format(channel_name, epoch))

    wandb.finish()

if __name__ == "__main__":
    main()