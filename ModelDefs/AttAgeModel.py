import json

import torch
import torch.nn as nn
import os

from ModelDefs.ChannelEncoderCNVGRU import EEGConvGRUEncoder


class AgeModel(nn.Module):
    def __init__(self, attention_heads=1,
                 common_dim=32,
                 pretrained_dir="ContrastiveModel_chkpoints",
                 hyperparam_config="../FindOptimalProbForAugmentation/contrastive_hyperparams.json",
                 unlock_encoders=True,
                 config_file='../config.json'):
        super(AgeModel, self).__init__()
        self.unlock_encoders = unlock_encoders

        with open(hyperparam_config, "r") as f:
            hyperparam_config_dict = json.load(f)
        with open(config_file, "r") as f:
            config = json.load(f)
        self.common_dim = common_dim
        self.eeg_columns = config['EEG_columns']

        # Load 1 pretrained ChannelEncoder per channel and freeze them
        self.encoders = nn.ModuleList()
        # One projection layer per channel to map encoder output -> common_dim
        self.channel_projectors = nn.ModuleList()

        for channel_name in config['EEG_columns']:
            hyperparams = hyperparam_config_dict[channel_name]
            hidden_size = hyperparams['hidden_size']
            num_layers_gru = hyperparams['num_layers_gru']
            projection_size = hyperparams['projection_size']
            cnn_dropout = hyperparams['CNN_dropout']
            rnn_dropout = hyperparams['RNN_dropout']
            activation = hyperparams['activation_function']
            rnn_type = hyperparams['rnn_type']
            kernel_size = hyperparams['kernel_size']
            stride = hyperparams['stride']
            padding = hyperparams['padding']
            channels = hyperparams['Conv1DChannels']

            model = EEGConvGRUEncoder(
                gru_hidden=hidden_size,
                embedding_dim=projection_size,
                num_layers_gru=num_layers_gru,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                cnn_dropout=cnn_dropout,
                activation=activation,
                conv_channels=channels,
                rnn_dropout=rnn_dropout,
                rnn_type=rnn_type
            )

            path = os.path.join(pretrained_dir, f"EEGContrsative_forCH_{channel_name}.pth")
            print(path)
            if os.path.exists(path):
                print(f"Loading pretrained ChannelEncoder for channel {channel_name} from {path}")
                checkpoint = torch.load(path, map_location=torch.device('cuda'))
                state_dict = checkpoint

                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
            else:
                print()
                raise FileNotFoundError(f"Missing pretrained model at {path}")

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = unlock_encoders

            self.encoders.append(model)
            self.channel_projectors.append(
                nn.Sequential(
                    nn.Linear(hidden_size, common_dim, bias=False),
                    nn.LayerNorm(common_dim)
                )
            )

        if common_dim % attention_heads != 0:
            raise ValueError(
                f"common_dim ({common_dim}) must be divisible by attention_heads ({attention_heads})"
            )
        # Self-attention across channels (not time!)
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=attention_heads, batch_first=True)

        # Dense layers
        self.a = nn.SiLU() #GELU()
        self.fc1 = nn.Linear(len(self.eeg_columns) * common_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.d = nn.Dropout(0.35)
        self.output = nn.Linear(256, 1)  # Regression output

    def train(self, mode=True):
        super().train(mode)
        if not self.unlock_encoders:
            # Frozen encoders: keep BN in eval mode to preserve pretrained running stats
            for enc in self.encoders:
                enc.eval()
        return self

    def forward(self, x):
        # x: [batch, seq_len, 14]
        features = []
        # Apply each pretrained encoder to its respective channel
        for i in range(len(self.eeg_columns)):
            channel_input = x[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            projection,f = self.encoders[i](channel_input)
            # Ensure encoder output is [B, D]
            if f.dim() == 3:
                if f.size(1) == 1:
                    f = f.squeeze(1)
                elif f.size(-1) == 1:
                    f = f.squeeze(-1)

            f = self.channel_projectors[i](f)

            # Ensure projected output is [B, common_dim]
            if f.dim() == 3:
                if f.size(1) == 1:
                    f = f.squeeze(1)
                elif f.size(-1) == 1:
                    f = f.squeeze(-1)

            if f.dim() != 2:
                raise ValueError(f"Expected per-channel feature [B, D], got {f.shape}")

            features.append(f)

        # Stack features
        channel_features = torch.stack(features, dim=1)

        # Apply self-attention across channels
        attn_output, attn_weights = self.attention(
            channel_features, channel_features, channel_features,
            need_weights=True, average_attn_weights=True
        )

        # Flatten and pass through dense layers
        out = attn_output.reshape(attn_output.size(0), -1)
        out = self.a(self.fc1(out))
        out = self.a(self.fc2(out))
        out = self.output(out)
        return out.squeeze(1), attn_weights
