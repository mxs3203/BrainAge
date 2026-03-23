import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGIndexedDataset(Dataset):
    def __init__(self, index_df, channels="all", target="Age", transform=None):
        self.index_df = index_df.reset_index(drop=True)
        self.channels = channels
        self.target = target
        self.transform = transform

        self.channel_map = {
            "EEG.AF3": 0,
            "EEG.F7": 1,
            "EEG.F3": 2,
            "EEG.FC5": 3,
            "EEG.T7": 4,
            "EEG.P7": 5,
            "EEG.O1": 6,
            "EEG.O2": 7,
            "EEG.P8": 8,
            "EEG.T8": 9,
            "EEG.FC6": 10,
            "EEG.F4": 11,
            "EEG.F8": 12,
            "EEG.AF4": 13,
        }

    def __len__(self):
        return len(self.index_df)

    def _load_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        sample = self._load_pickle(row["file_path"])

        X = sample["X"]  # expected (512, 14)

        if not isinstance(X, np.ndarray):
            raise ValueError(f"X is not numpy array in {row['file_path']}")

        if X.shape != (512, 14):
            raise ValueError(f"Unexpected X shape {X.shape} in {row['file_path']}")

        X = torch.tensor(X, dtype=torch.float32)

        y = None
        if self.target is not None and self.target in sample:
            y = torch.tensor(float(sample[self.target]), dtype=torch.float32)

        if self.transform:
            out = self.transform(X)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out
            X = out

        return (X, y) if y is not None else X