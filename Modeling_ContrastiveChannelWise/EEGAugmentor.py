import random

import torch


class EEGAugmentor:
    def __init__(self):
        pass

    def __call__(self, x, prob):
        # x: [seq_len, channels]
        if random.random() < 0.7:
            x = self.jitter(x)
        if random.random() < 0.5:
            x = self.scaling(x)
        if random.random() < 0.3:
            x = self.time_mask(x)
        return x

    def jitter(self, x, sigma=0.01):
        return x + sigma * torch.randn_like(x)

    def scaling(self, x, sigma=0.1):
        factor = torch.normal(mean=1.0, std=sigma, size=(1, x.size(1))).to(x.device)
        return x * factor

    def time_mask(self, x, max_mask_pct=0.1):
        seq_len = x.size(0)
        mask_len = int(seq_len * max_mask_pct)
        start = torch.randint(0, seq_len - mask_len, (1,)).item()
        x[start:start+mask_len] = 0
        return x

    def channel_dropout(self, x, drop_prob=0.2):
        mask = (torch.rand(x.size(1)) > drop_prob).float().to(x.device)
        return x * mask

