import glob

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from tqdm import tqdm


class EEGDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for subject_folder in tqdm(glob.glob(root_dir + '/*')):
            path_to_pck = os.path.join(subject_folder, "7_lstm_rdy_data.pck")
            if os.path.exists(path_to_pck):
                with open(path_to_pck, "rb") as f:
                    subject_data = pickle.load(f)
                    self.samples.extend(subject_data)  # each entry is a dict with 'data' and 'Age'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample["data"], dtype=torch.float32)  # shape: (window_size, 14)
        y = torch.tensor(sample["Age"], dtype=torch.float32)  # single float

        return x, y

dataset = EEGDataset(root_dir="../results")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
for batch_x, batch_y in dataloader:
    print(batch_x.shape)  # (batch_size, window_size, 14)
    print(batch_y.shape)  # (batch_size,)
    break  # just to inspect one batch
