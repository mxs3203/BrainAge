import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from Modeling.BasicMultiChannelLSTM import EEG_LSTM
from Modeling.EEGDataset import EEGDataset
from Modeling.SingleCH_LSTM_Att_model import ChannelWiseLSTMModel
from utils.Visualization import plot_attention_heatmaps
import seaborn as sns

dataset = EEGDataset(root_dir="../results")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


LR = 1e-4
hidden_size = 16
num_lstm_layers = 4
num_epochs = 400
BATCH_SIZE = 512
num_att_heads = 8


train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, optimizer
#model = EEG_LSTM(input_size=14, hidden_size=hidden_size, num_layers=num_lstm_layers).to(device)
model = ChannelWiseLSTMModel(hidden_size=hidden_size, attention_heads=num_att_heads, num_layers=num_lstm_layers)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
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
        "num_lstm_layers": num_lstm_layers,
        "num_attention_heads": num_att_heads,
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
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "EEGModel_{}.pth".format(epoch))
        plot_attention_heatmaps(epoch=epoch, attn_weights=att, output_var=y)

wandb.finish()