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
            kernel_size = hyperparams['kernel_size']
            stride = hyperparams['stride']
            padding = hyperparams['padding']
            channels = hyperparams['channels']

            model = EEGConvGRUEncoder(
                hidden_size=hidden_size,
                gru_hidden=hidden_size,
                embedding_dim=projection_size,
                num_layers_gru=num_layers_gru,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                conv_channels=channels
            )

            path = os.path.join(pretrained_dir, f"EEGContrsative_forCH_{channel_name}_Epoch1000.pth")
            print(path)
            if os.path.exists(path):
                print(f"Loading pretrained ChannelEncoder for channel {channel_name} from {path}")
                checkpoint = torch.load(path, map_location=torch.device('cuda'))
                state_dict = checkpoint

                # If model was trained with DataParallel, the keys will have 'module.' prefix
                if any(k.startswith('module.') for k in state_dict.keys()):
                    print(f"Detected 'module.' prefix in the state_dict. Removing it...")
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                #model.load_state_dict(state_dict)
            else:
                print()
                raise FileNotFoundError(f"Missing pretrained model at {path}")

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = unlock_encoders

            self.encoders.append(model)
            self.channel_projectors.append(
                nn.Sequential(
                    nn.Linear(projection_size, common_dim, bias=False),
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
        self.fc1 = nn.Linear(len(self.eeg_columns) * common_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.d = nn.Dropout(0.25)
        self.output = nn.Linear(32, 1)  # Regression output

    def forward(self, x):
        # x: [batch, seq_len, 14]
        features = []
        # Apply each pretrained encoder to its respective channel
        for i in range(len(self.eeg_columns)):
            channel_input = x[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            f = self.encoders[i](channel_input)
            f = self.channel_projectors[i](f)
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
