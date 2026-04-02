import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class SupervisedContrastiveLearningDataset(Dataset):
    """
    Dataset for supervised contrastive learning with continuous age labels.

    Each __getitem__ returns two augmented views of a single EEG window
    plus the subject's age. The loss function (supervised_contrastive_loss)
    then uses the age to define positives within the batch: all windows
    from subjects within `age_threshold` years are pulled together.

    Unlike EEGContrastivePairs (same subject + same stimulus) or
    EEGCrossStimulusContrastivePairs (same subject + different stimulus),
    this dataset imposes NO subject-identity constraint — two windows from
    completely different subjects can be positives if their ages are close.
    This directly aligns the pretraining objective with the downstream task.

    index structure:
        { ID -> { stimulus -> [file_path, ...] } }
    """

    def __init__(self, data_dir, channel_idx=0, subset_fraction=1.0, seed=42,
                 cache_name="supervised_contrastive_index_cache.pkl", preload=True):
        self.data_dir = data_dir
        self.channel_idx = channel_idx
        self.rng = random.Random(seed)
        self.cache_path = os.path.join(data_dir, cache_name)

        # flat list of (file_path, age) for all valid windows
        self.samples = self._build_index()

        if not self.samples:
            raise RuntimeError(f"No valid .pkl windows found in {data_dir}")

        if subset_fraction < 1.0:
            self.samples = self._stratified_sample(self.samples, subset_fraction)

        ages = [age for _, age in self.samples]
        print(
            f"[SupervisedContrastiveLearningDataset] {len(self.samples)} windows | "
            f"channel_idx={channel_idx} | "
            f"age range {min(ages):.0f}–{max(ages):.0f}"
        )

        self.preload = preload
        if preload:
            # Pre-load all windows into RAM to eliminate per-sample file I/O.
            # At subset_fraction=0.1 this is ~1.8 GB per channel — fast array
            # indexing in __getitem__ instead of open+unpickle on every call.
            # For larger datasets set preload=False to read from disk on demand.
            print("[SupervisedContrastiveLearningDataset] Pre-loading windows into RAM...")
            n = len(self.samples)
            self.data    = np.empty((n, 512), dtype=np.float32)
            self.ages_arr = np.empty(n,        dtype=np.float32)
            self.ids     = []
            self.stimuli = []
            for i, (path, age) in enumerate(self.samples):
                with open(path, "rb") as f:
                    s = pickle.load(f)
                self.data[i]     = s["X"][:, channel_idx]
                self.ages_arr[i] = age
                self.ids.append(s.get("ID", ""))
                self.stimuli.append(s.get("stimulus", ""))
            self.samples = None  # free path list — no longer needed
            print(f"[SupervisedContrastiveLearningDataset] Loaded. "
                  f"RAM footprint: {self.data.nbytes / 1e9:.2f} GB")

    @staticmethod
    def _age_bin(age):
        """Coarse 5-year bins covering the dataset range (20–65)."""
        return int(age // 5) * 5

    def _stratified_sample(self, samples, fraction):
        """
        Sample `fraction` of windows while preserving the age distribution.
        Windows are grouped into 5-year age bins; `fraction` is taken from
        each bin independently so under-represented age groups are not lost.
        """
        bins = defaultdict(list)
        for item in samples:
            bins[self._age_bin(item[1])].append(item)

        result = []
        for bin_label, bin_samples in sorted(bins.items()):
            k = max(1, int(len(bin_samples) * fraction))
            chosen = self.rng.sample(bin_samples, k)
            result.extend(chosen)

        n_total = len(samples)
        print(
            f"[SupervisedContrastiveLearningDataset] Stratified subset: "
            f"{len(result)}/{n_total} windows ({len(result)/n_total:.1%}) "
            f"across {len(bins)} age bins."
        )
        return result

    def _build_index(self):
        """
        Returns a flat list of (file_path, age) tuples.
        Uses a cache to avoid re-scanning millions of files.
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                samples = pickle.load(f)
            if isinstance(samples, list):
                print(f"[SupervisedContrastiveLearningDataset] Loaded cached index ({len(samples)} windows).")
                return samples

        print("[SupervisedContrastiveLearningDataset] Building index (first run only)...")
        samples = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".pkl"):
                continue
            path = os.path.join(self.data_dir, fname)
            try:
                with open(path, "rb") as f:
                    s = pickle.load(f)
                age = s.get("Age", None)
                if age is None or (isinstance(age, float) and age != age):  # skip NaN
                    continue
                samples.append((path, float(age)))
            except Exception as e:
                print(f"  Skip {fname}: {e}")

        with open(self.cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"[SupervisedContrastiveLearningDataset] Cached index saved ({len(samples)} windows).")
        return samples

    def augment(self, x):
        x = x.clone()

        noise_std = 0.01 * x.std()
        x = x + torch.randn_like(x) * noise_std

        scale = 1.0 + self.rng.uniform(-0.1, 0.1)
        x = x * scale

        shift = self.rng.randint(-5, 5)
        if shift > 0:
            x = torch.cat([torch.zeros_like(x[:shift]), x[:-shift]], dim=0)
        elif shift < 0:
            x = torch.cat([x[-shift:], torch.zeros_like(x[:(-shift)])], dim=0)

        drop_len = self.rng.randint(0, 20)
        if drop_len > 0:
            start = self.rng.randint(0, max(1, len(x) - drop_len + 1))
            x[start:start + drop_len] = 0.0

        return x

    def __len__(self):
        return len(self.data) if self.preload else len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            x   = torch.from_numpy(self.data[idx].copy())
            age = float(self.ages_arr[idx])
            meta = {"ID": self.ids[idx], "stimulus": self.stimuli[idx], "Age": age}
        else:
            path, age = self.samples[idx]
            with open(path, "rb") as f:
                s = pickle.load(f)
            x = torch.as_tensor(s["X"][:, self.channel_idx], dtype=torch.float32)
            meta = {"ID": s.get("ID", ""), "stimulus": s.get("stimulus", ""), "Age": age}

        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1, x2, torch.tensor(age, dtype=torch.float32), meta
