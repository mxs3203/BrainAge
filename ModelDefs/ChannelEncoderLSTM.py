import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelEncoder(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers, proj_dim, dropout=0.2):
        super(ChannelEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.projection_head = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_size, proj_dim)
        )

    def forward(self, x):

        if x.dim() == 2:
            # [B, L] → [B, L, 1]
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.shape[1] > x.shape[2]:
            # [B, C, L] → [B, L, C]
            x = x.permute(0, 2, 1).contiguous()
        elif x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, L] or [B, L, C] or [B, C, L], got {x.shape}")

        out, _ = self.lstm(x)               # [B, L, 2*hidden]
        out = out.mean(dim=1)               # temporal average pooling
        z = self.projection_head(out)       # [B, proj_dim]
        return F.normalize(z, dim=1)
