import torch
import torch.nn as nn


class EEGAugmentorGPU(nn.Module):
    """
    GPU-side augmentation for 1D EEG windows.

    Operates on batches of shape [B, T] already on the GPU, replacing the
    per-sample CPU augmentation in __getitem__. Running augmentation on the
    GPU eliminates the CPU bottleneck when data is preloaded into RAM.

    Augmentations (all vectorised over the batch dimension):
      - Gaussian noise scaled to the signal's own std
      - Random amplitude scaling
      - Random time shift (circular roll)
      - Random segment dropout (zeroing a contiguous block)
    """

    def __init__(self,
                 noise_std_factor: float = 0.01,
                 scale_range: float = 0.1,
                 max_shift: int = 5,
                 max_drop_len: int = 20):
        super().__init__()
        self.noise_std_factor = noise_std_factor
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.max_drop_len = max_drop_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] float32 tensor on GPU
        Returns:
            augmented [B, T] tensor, same device
        """
        B, T = x.shape

        # 1. Gaussian noise — scale per sample to its own std
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x = x + torch.randn_like(x) * (self.noise_std_factor * std)

        # 2. Amplitude scaling — one scalar per sample
        scale = 1.0 + (torch.rand(B, 1, device=x.device) * 2 - 1) * self.scale_range
        x = x * scale

        # 3. Time shift — different roll per sample via scatter
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=x.device)
        indices = torch.remainder(torch.arange(T, device=x.device).unsqueeze(0) - shifts.unsqueeze(1), T)
        x = x.gather(1, indices)

        # 4. Segment dropout — zero a random contiguous block per sample
        if self.max_drop_len > 0:
            drop_lens  = torch.randint(0, self.max_drop_len + 1, (B,), device=x.device)
            drop_starts = torch.randint(0, T, (B,), device=x.device)
            t_idx = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            # mask[b, t] = 1 if t is inside the dropout window for sample b
            mask = (t_idx >= drop_starts.unsqueeze(1)) & \
                   (t_idx < (drop_starts + drop_lens).unsqueeze(1))
            x = x * (~mask).float()

        return x
