import glob
import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Modeling_ContrastiveChannelWise.EEGAugmentor import EEGAugmentor


class EEGContrastiveDataset(Dataset):
    def __init__(self, root_dir, channel_idx, prob_aug=0.3, contrastive=True, all_channels=True):
        self.samples = []
        self.channel_idx = channel_idx
        self.contrastive = contrastive
        self.all_channels = all_channels
        self.prob_aug = prob_aug
        self.augment = EEGAugmentor()

        print("Preloading entire data into memory...")
        for subject_folder in tqdm(glob.glob(root_dir + '/*')):
            path_to_pck = os.path.join(subject_folder, "7_lstm_rdy_data.pck")
            if os.path.exists(path_to_pck):
                with open(path_to_pck, "rb") as f:
                    subject_data = pickle.load(f)
                    self.samples.extend(subject_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample["data"], dtype=torch.float32)  # shape: (window, channels)

        if not self.all_channels:
            channel_idx = self.channel_idx
            x = x[:, channel_idx].unsqueeze(1)  # shape: (window, 1)

        if self.contrastive:
            x1 = self.augment(x.clone(), prob=self.prob_aug)
            x2 = self.augment(x.clone(), prob=self.prob_aug)
            return x1, x2
        else:
            y = torch.tensor(sample["Age"], dtype=torch.float32)
            return x, y