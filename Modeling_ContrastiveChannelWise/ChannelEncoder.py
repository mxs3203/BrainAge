import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelEncoder(nn.Module):
    def __init__(self, input_size=1, lstm_hidden_size=64, lstm_layers=1, proj_dim=128):
        super(ChannelEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.projection_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, proj_dim)
        )

    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)           # hn: [num_layers, batch, hidden_size]
        features = hn[-1]                          # take last layer's hidden state → [batch, hidden_size]
        proj = self.projection_head(features)      # [batch, proj_dim]
        proj = F.normalize(proj, dim=1)        # normalize for contrastive loss
        return proj
