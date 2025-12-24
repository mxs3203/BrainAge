import glob
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



class EEGDataset(Dataset):
    def __init__(self, root_dir, channels="all", target="Age", transform=None, preload=False, subset_size=None,seed=27):
        self.root_dir = root_dir
        self.pkl_files = []
        self.pkl_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pkl")]
        self.original_size = self.pkl_files
        if not self.pkl_files:
            raise ValueError(f"No .pkl files found in {root_dir}")

        self.target = target
        self.transform = transform
        self.preload = preload
        self.channel_idx = channels

        sample = self._load_pickle(self.pkl_files[0])
        self.all_eeg_cols = [c for c in sample.keyscd () if c.startswith("EEG.")] \
            if isinstance(sample, dict) and "X" not in sample else [f"EEG.{i}" for i in range(sample["X"].shape[1])]

        if subset_size is not None:
            random.seed(seed)
            random.shuffle(self.pkl_files)
            self.pkl_files = self.pkl_files[:subset_size]
            print(f"Using subset of {subset_size}/{len(self.pkl_files)} windows from {root_dir}")
        else:
            self.pkl_files = self.pkl_files
            print(f"Loaded full dataset with {len(self.pkl_files)} windows from {root_dir}")

        if preload:
            print("Preloading entire data into memory...")
            self.data = [self._load_pickle(fp) for fp in tqdm(self.pkl_files)]
        else:
            self.data = None

    def _load_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.original_size)

    def __getitem__(self, idx):
        sample = self.data[idx] if self.data else self._load_pickle(self.pkl_files[idx])
        X = sample["X"]  # numpy array (window_size, num_channels)
        if isinstance(X, np.ndarray):
            if X.shape != (512, 14):
                self.__getitem__(np.random.randint(0, len(self)))
        else:
            self.__getitem__(np.random.randint(0, len(self)))

        # --- Channel selection ---
        if self.channel_idx != "all":
            if not isinstance(self.channel_idx, int):
                raise ValueError("channel_idx must be 'all' or an integer.")
            if self.channel_idx >= X.shape[1]:
                raise IndexError(f"Channel index {self.channel_idx} exceeds available channels {X.shape[1]}")
            X = X[:, [self.channel_idx]]  # (T, 1)

        X = torch.tensor(X, dtype=torch.float32)

        # --- Target ---
        y = None
        if self.target and self.target in sample:
            val = sample[self.target]
            if isinstance(val, (int, float)):
                y = torch.tensor(val, dtype=torch.float32)
            elif isinstance(val, str):
                y = val  # keep as string label

        if self.transform:
            out = self.transform(X)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out  # contrastive mode
            else:
                X = out  # standard mode

        y = None
        if self.target and self.target in sample:
            val = sample[self.target]
            y = torch.tensor(val, dtype=torch.float32) if isinstance(val, (int, float)) else val

        return (X, y) if y is not None else X
