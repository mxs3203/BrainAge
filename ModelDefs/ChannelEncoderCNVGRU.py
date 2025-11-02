import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGConvGRUEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, gru_hidden=128, embedding_dim=64):
        super().__init__()

        # --- Convolutional feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),  # [B, 32, L/2]
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, hidden_size, kernel_size=3, stride=2, padding=1),     # [B, H, L/8]
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # --- GRU over time (sequence dimension after conv) ---
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Projection head (embedding for contrastive learning) ---
        self.projector = nn.Sequential(
            nn.Linear(gru_hidden * 2, embedding_dim),
            nn.ReLU(),
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
        out, _ = self.gru(feats)                       # [B, L', 2*gru_hidden]
        out = out.mean(dim=1)                          # global average pooling over time

        # Projection head
        emb = self.projector(out)                      # [B, embedding_dim]
        emb = F.normalize(emb, dim=1)                  # normalize for cosine similarity

        return emb