import torch
import torch.nn as nn
import os

from Modeling_ContrastiveChannelWise.ChannelEncoder import ChannelEncoder


class ChannelWiseLSTMContrastiveModel(nn.Module):
    def __init__(self, hidden_size=64, attention_heads=1, fc_hidden=64, projection_head_size=16, num_layers=1, pretrained_dir="ContrastiveModel_chkpoints"):
        super(ChannelWiseLSTMContrastiveModel, self).__init__()

        self.num_channels = 14
        self.hidden_size = hidden_size

        # Load 1 pretrained ChannelEncoder per channel and freeze them
        self.encoders = nn.ModuleList()
        for i in range(self.num_channels):
            encoder = ChannelEncoder(lstm_hidden_size=hidden_size, lstm_layers=num_layers, proj_dim=projection_head_size)
            path = os.path.join(pretrained_dir, f"EEGContrsative_forCH_{i}_Epoch40.pth")
            print(path)
            if os.path.exists(path):
                print(f"Loading pretrained ChannelEncoder for channel {i} from {path}")
                checkpoint = torch.load(path, map_location=torch.device('cuda'))
                state_dict = checkpoint

                # If model was trained with DataParallel, the keys will have 'module.' prefix
                if any(k.startswith('module.') for k in state_dict.keys()):
                    print(f"⚠️ Detected 'module.' prefix in the state_dict. Removing it...")
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                encoder.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Missing pretrained model at {path}")

            # Freeze all parameters
            for param in encoder.parameters():
                param.requires_grad = False

            self.encoders.append(encoder)

        # Self-attention across channels (not time!)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size * self.num_channels, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)  # Regression output

    def forward(self, x):
        # x: [batch, seq_len, 14]
        batch_size = x.size(0)
        features = []

        # Apply each pretrained encoder to its respective channel
        for i in range(self.num_channels):
            channel_input = x[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            feature = self.encoders[i].lstm(channel_input)[1][0][-1]  # use raw LSTM output
            features.append(feature)

        # Stack features: [batch, 14, hidden_size]
        channel_features = torch.stack(features, dim=1)

        # Apply self-attention across channels
        attn_output, attn_weights = self.attention(
            channel_features, channel_features, channel_features,
            need_weights=True, average_attn_weights=False
        )
        attn_weights = attn_weights.mean(dim=1)

        # Flatten and pass through dense layers
        out = attn_output.reshape(batch_size, -1)  # [batch, 14 * hidden_size]
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.output(out)
        return out.squeeze(1), attn_weights
