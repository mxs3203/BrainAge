import glob

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

from tqdm import tqdm


class LazyEEGDataset(Dataset):
    def __init__(self, root_dir):

        self.index = []  # list of tuples: (filepath, sample_index)

        for subject_folder in tqdm(glob.glob(root_dir + '/*')):
            pck_path = os.path.join(subject_folder, "7_lstm_rdy_data.pck")
            if os.path.exists(pck_path):
                with open(pck_path, "rb") as f:
                    data = pickle.load(f)
                    for i in range(len(data)):
                        self.index.append((pck_path, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pck_path, sample_idx = self.index[idx]
        with open(pck_path, "rb") as f:
            samples = pickle.load(f)
            sample = samples[sample_idx]

        x = torch.tensor(sample["data"], dtype=torch.float32)  # shape: (window_size, 14)
        y = torch.tensor(sample["Age"], dtype=torch.float32)   # scalar

        return x, y
