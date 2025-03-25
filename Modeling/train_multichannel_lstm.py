import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from Modeling.BasicMultiChannelLSTM import EEG_LSTM
from Modeling.EEGDataset import EEGDataset
from Modeling.SingleCH_LSTM_Att_model import ChannelWiseLSTMModel

dataset = EEGDataset(root_dir="../results")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


LR = 1e-4
hidden_size = 64
num_lstm_layers = 1
num_epochs = 50
BATCH_SIZE = 512


train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, optimizer
#model = EEG_LSTM(input_size=14, hidden_size=hidden_size, num_layers=num_lstm_layers).to(device)
model = ChannelWiseLSTMModel(hidden_size=hidden_size, attention_heads=8, num_layers=num_lstm_layers).to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=LR)


wandb.init(
    project="BrainAge",
    entity="mxs3203",
    config={
        "model": "ChannelWiseLSTMAtt",
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": LR,
        "hidden_size": hidden_size,
        "input_size": 14,
        "num_lstm_layers": num_lstm_layers
    }
)

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_train_loss = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs, att = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            outputs, att = model(x)
            loss = criterion(outputs, y)
            total_val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(val_loader):.4f}")
    wandb.log({
        "Train Loss": total_train_loss/len(train_loader),
        "Validation Loss": total_val_loss/len(val_loader),
        "Epoch": epoch + 1
    })
wandb.finish()