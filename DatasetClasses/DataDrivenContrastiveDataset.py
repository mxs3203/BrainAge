import os, pickle, random
from collections import defaultdict
from torch.utils.data import Dataset
import torch

class EEGContrastivePairs(Dataset):
    def __init__(self, data_dir, channel_idx=0, subset_fraction=1.0, seed=42, cache_name="index_cache.pkl"):
        self.data_dir = data_dir
        self.channel_idx = channel_idx
        self.cache_path = os.path.join(data_dir, cache_name)
        self.rng = random.Random(seed)

        # Build/load index: (ID, stimulus) -> [file paths]
        index = self._build_index()
        # Keep only groups with >=2 windows
        self.groups = [(k, v) for k, v in index.items() if len(v) >= 2]

        if not self.groups:
            raise RuntimeError("No (ID, stimulus) groups with ≥2 windows. Cannot form positive pairs.")

        # Optional subsetting by groups (not by files)
        if subset_fraction < 1.0:
            n = len(self.groups)
            k = max(1, int(n * subset_fraction))
            self.groups = self.rng.sample(self.groups, k)
            print(f"[EEGContrastivePairs] Using {k}/{n} groups ({k/n:.1%}).")

    def _build_index(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                idx = pickle.load(f)
            if isinstance(idx, dict):
                print(f"[EEGContrastivePairs] Loaded cached index ({len(idx)} groups).")
                return idx

        idx = defaultdict(list)
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".pkl"):
                continue
            p = os.path.join(self.data_dir, fname)
            try:
                with open(p, "rb") as f:
                    s = pickle.load(f)
                idx[(s["ID"], s["stimulus"])].append(p)
            except Exception as e:
                print(f"Skip {fname}: {e}")

        with open(self.cache_path, "wb") as f:
            pickle.dump(idx, f)
        print(f"[EEGContrastivePairs] Cached index saved ({len(idx)} groups).")
        return idx

    def __len__(self):
        # Length = number of groups; each __getitem__ draws a fresh pair from that group
        return len(self.groups)

    def __getitem__(self, idx):
        (ID, stim), files = self.groups[idx]
        f1, f2 = self.rng.sample(files, 2)

        with open(f1, "rb") as f:
            s1 = pickle.load(f)
        with open(f2, "rb") as f:
            s2 = pickle.load(f)

        # X is (T, C); take single channel -> (T,)
        x1 = torch.as_tensor(s1["X"][:, self.channel_idx], dtype=torch.float32)
        x2 = torch.as_tensor(s2["X"][:, self.channel_idx], dtype=torch.float32)

        # meta kept lightweight & collate-friendly
        meta = {"ID": ID, "stimulus": stim,
                "Age1": float(s1.get("Age", float("nan"))),
                "Age2": float(s2.get("Age", float("nan")))}

        return x1, x2, meta
