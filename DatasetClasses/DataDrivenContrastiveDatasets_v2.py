import os
import pickle
import random
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
import torch

'''

    Positive pairs belong to the same stimuli but not the same ID
'''

class EEGContrastivePairsSameStimDifferentID(Dataset):
    def __init__(self, data_dir, channel_idx=0, subset_fraction=1.0, seed=42,
                 cache_name="index_cache_same_stim_diff_id.pkl"):
        self.data_dir = data_dir
        self.channel_idx = channel_idx
        self.cache_path = os.path.join(data_dir, cache_name)
        self.rng = random.Random(seed)
        stim_names = pd.read_csv("../stimuli_descriptions_and_expectations.csv")
        allowed_stimuli = (
            stim_names["original_stimuli_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        # Build/load index: stimulus -> ID -> [file paths]
        index = self._build_index()
        index = {
            k: v for k, v in index.items()
            if (k[1] if isinstance(k, tuple) else k) in allowed_stimuli
        }
        # Keep only stimuli that have at least 2 different IDs
        self.groups = []
        for stim, id_dict in index.items():
            valid_ids = [ID for ID, files in id_dict.items() if len(files) >= 1]
            if len(valid_ids) >= 2:
                self.groups.append((stim, {ID: id_dict[ID] for ID in valid_ids}))

        if not self.groups:
            raise RuntimeError(
                "No stimulus groups with at least 2 different IDs. Cannot form positive pairs."
            )

        # Optional subsetting by stimulus groups
        if subset_fraction < 1.0:
            n = len(self.groups)
            k = max(1, int(n * subset_fraction))
            self.groups = self.rng.sample(self.groups, k)
            print(f"[EEGContrastivePairsSameStimDifferentID] Using {k}/{n} stimulus groups ({k/n:.1%}).")

    def _build_index(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                idx = pickle.load(f)
            if isinstance(idx, dict):
                print(f"[EEGContrastivePairsSameStimDifferentID] Loaded cached index ({len(idx)} stimuli).")
                return idx

        idx = defaultdict(lambda: defaultdict(list))

        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".pkl"):
                continue

            p = os.path.join(self.data_dir, fname)
            try:
                with open(p, "rb") as f:
                    s = pickle.load(f)

                stim = s["stimulus"]
                ID = s["ID"]
                idx[stim][ID].append(p)

            except Exception as e:
                print(f"Skip {fname}: {e}")

        # convert nested defaultdict to normal dict before pickling
        idx = {stim: dict(id_dict) for stim, id_dict in idx.items()}

        with open(self.cache_path, "wb") as f:
            pickle.dump(idx, f)

        print(f"[EEGContrastivePairsSameStimDifferentID] Cached index saved ({len(idx)} stimuli).")
        return idx

    def __len__(self):
        # Length = number of valid stimuli
        return len(self.groups)

    def __getitem__(self, idx):
        stim, id_dict = self.groups[idx]

        # sample 2 different IDs for the same stimulus
        id1, id2 = self.rng.sample(list(id_dict.keys()), 2)

        # sample one file from each ID
        f1 = self.rng.choice(id_dict[id1])
        f2 = self.rng.choice(id_dict[id2])

        with open(f1, "rb") as f:
            s1 = pickle.load(f)
        with open(f2, "rb") as f:
            s2 = pickle.load(f)

        # X is (T, C); take single channel -> (T,)
        x1 = torch.as_tensor(s1["X"][:, self.channel_idx], dtype=torch.float32)
        x2 = torch.as_tensor(s2["X"][:, self.channel_idx], dtype=torch.float32)

        meta = {
            "ID1": id1,
            "ID2": id2,
            "stimulus": stim,
            "Age1": float(s1.get("Age", float("nan"))),
            "Age2": float(s2.get("Age", float("nan")))
        }

        return x1, x2, meta