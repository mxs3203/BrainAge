import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGConvGRUEncoder(nn.Module):
    def __init__(self, input_channels=1,
                 num_layers_gru=1,gru_hidden=12, embedding_dim=64,
                 kernel_size=7, stride=2, padding=1, conv_channels=32,
                 activation="ReLU", rnn_dropout=0.0, rnn_type="GRU",
                 ):
        super().__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "SiLU":
            self.activation = nn.SiLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        self.rnn_dropout = rnn_dropout
        # --- Convolutional feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels, kernel_size=kernel_size, stride=stride, padding=padding),  # [B, 32, L/2]
            nn.BatchNorm1d(conv_channels),
            self.activation,

            # nn.Conv1d(conv_channels, hidden_size, kernel_size=kernel_size2, stride=stride2, padding=padding2),     # [B, H, L/8]
            # nn.BatchNorm1d(hidden_size),
            # nn.GELU(),
        )
        hidden_size = conv_channels
        # --- GRU over time (sequence dimension after conv) ---
        if rnn_type == "GRU":
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=gru_hidden,
                num_layers=num_layers_gru,
                dropout=self.rnn_dropout,
                batch_first=True,
                bidirectional=False
            )
        elif rnn_type == "LSTM":
            self.gru = nn.LSTM(
                input_size=hidden_size,
                hidden_size=gru_hidden,
                num_layers=num_layers_gru,
                dropout=self.rnn_dropout,
                batch_first=True,
                bidirectional=False
            )
        else:
            print("Unknown RNN type")
        gru_out_dim = gru_hidden * 2 if self.gru.bidirectional else gru_hidden
        # --- Projection head (embedding for contrastive learning) ---
        self.projector = nn.Sequential(
            nn.Linear(gru_out_dim, embedding_dim),
            self.activation,
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        """
        x: [B, 1, L] where L = time steps (e.g., 256 or 512)
        """
        if x.dim() == 2:
            # [B, L] → [B, 1, L]
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] > x.shape[2]:
            # [B, L, C] → [B, C, L]
            x = x.permute(0, 2, 1).contiguous()
        else:
            assert x.dim() == 3, f"Expected 3D tensor, got {x.shape}"
        # CNN expects [B, C, L]
        feats = self.conv(x)                           # [B, hidden, L']
        feats = feats.permute(0, 2, 1)                 # [B, L', hidden] for GRU

        # GRU temporal modeling
        out, h_n = self.gru(feats)                       # [B, L', 2*gru_hidden]
        #out = out.mean(dim=1)                          # global average pooling over time
        if self.gru.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # If bidiractional
        else:
            out = h_n[-1]
        # Projection head
        emb = self.projector(out)                      # [B, embedding_dim]
        emb = F.normalize(emb, dim=1)                  # normalize for cosine similarity

        return emb

class ModelWrapper(torch.nn.Module):
    def __init__(self, full_model, channel_idx):
        super().__init__()
        self.base_model = full_model.module if isinstance(full_model, torch.nn.DataParallel) else full_model
        self.channel_idx = channel_idx

    def forward(self, x):
        # x shape [B, L]
        x_full = torch.zeros(x.shape[0], x.shape[1], len(self.base_model.eeg_columns)).to(x.device)
        x_full[:,:,self.channel_idx] = x
        out, _ = self.base_model(x_full)
        return out