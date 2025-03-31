import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from Modeling_ContrastiveChannelWise.ChannelEncoder import ChannelEncoder
from Modeling_ContrastiveChannelWise.EEGContrastiveDataset import EEGContrastiveDataset
from utils.UnsupervisedContrastiveLoss import nt_xent_loss

channel_order = [
        "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7",
        "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
]
def train_channel(CH):
        LR = 1e-4
        hidden_size = 32
        num_lstm_layers = 1
        projection_size = 16
        num_epochs = 50 + 1
        BATCH_SIZE = 1024

        dataset = EEGContrastiveDataset(root_dir="../results", channel_idx=CH, contrastive=True, all_channels=False)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ChannelEncoder(lstm_hidden_size=hidden_size, lstm_layers=num_lstm_layers, proj_dim=projection_size)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)


        wandb.init(
            project="BrainAge",
            entity="mxs3203",
            name="Contrastive_Channel_{}".format(CH),
            config={
                "model": "Contrastive_SingleChannel{}".format(CH),
                "Channel": CH,
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": LR,
                "hidden_size": hidden_size,
                "input_size": 1,
                "num_lstm_layers": num_lstm_layers
            }
        )

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            model.train()
            total_train_loss = 0

            for i, (x1, x2) in enumerate(train_loader):
                x1, x2 = x1.to(device), x2.to(device)
                optimizer.zero_grad()
                z1,z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i, (x1, x2) in enumerate(val_loader):
                    x1, x2 = x1.to(device), x2.to(device)
                    optimizer.zero_grad()
                    z1, z2 = model(x1), model(x2)
                    loss = nt_xent_loss(z1, z2)
                    total_val_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(val_loader):.4f}")
            wandb.log({
                "Train Loss": total_train_loss/len(train_loader),
                "Validation Loss": total_val_loss/len(val_loader),
                "Epoch": epoch + 1
            })
            if epoch % 10 == 0:
                torch.save(model.state_dict(), "ContrastiveModel_chkpoints/EEGContrsative_forCH_{}_Epoch{}.pth".format(CH, epoch))

        wandb.finish()

def main():
    for CH in [8,9,10,11,12,13]:
        train_channel(CH)

if __name__ == "__main__":
    main()