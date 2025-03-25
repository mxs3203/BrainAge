import torch
import torch.nn as nn


class ChannelWiseLSTMModel(nn.Module):
    def __init__(self, hidden_size=32, attention_heads=1, fc_hidden=64, num_layers=1):
        super(ChannelWiseLSTMModel, self).__init__()

        self.num_channels = 14
        self.hidden_size = hidden_size

        # 1 LSTM per channel
        self.lstm_per_channel = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            for _ in range(self.num_channels)
        ])

        # Self-attention across channels (not time!)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size * self.num_channels, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)  # Regression output

    def forward(self, x):
        # x: [batch, seq_len, 14]
        batch_size = x.size(0)
        features = []

        # Apply LSTM to each channel independently
        for i in range(self.num_channels):
            channel_input = x[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            out, (hn, _) = self.lstm_per_channel[i](channel_input)  # hn: [1, batch, hidden]
            features.append(hn[-1])  # [batch, hidden_size]

        # Stack features: [batch, 14, hidden_size]
        channel_features = torch.stack(features, dim=1)

        # Apply self-attention across channels
        attn_output, _ = self.attention(channel_features, channel_features, channel_features)

        # Flatten and pass through dense layers
        out = attn_output.reshape(batch_size, -1)  # [batch, 14 * hidden_size]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(1), attn_output
