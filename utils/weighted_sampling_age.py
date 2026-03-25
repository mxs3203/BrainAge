import os
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def build_eeg_index(root_dir, output_csv=None, output_pkl=None):
    root_dir = Path(root_dir)
    pkl_files = sorted(root_dir.glob("*.pkl"))

    rows = []

    for file_path in tqdm(pkl_files, desc="Indexing EEG windows"):
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)

            row = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "ID": sample.get("ID", None),
                "Age": sample.get("Age", None),
            }

            rows.append(row)

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    df = pd.DataFrame(rows)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    if output_pkl is not None:
        df.to_pickle(output_pkl)

    return df

def split_index_by_participant(index_df, test_size=0.2, random_state=None):
    participant_df = (
        index_df
        .dropna(subset=["ID", "Age", "age_bin"])
        .groupby("ID", as_index=False)
        .agg({
            "Age": "first",
            "age_bin": "first"
        })
    )

    participant_df.sample(n=50, random_state=random_state)

    train_ids, val_ids = train_test_split(
        participant_df['ID'],
        test_size=test_size,
        random_state=random_state,
        stratify=participant_df["age_bin"]
    )

    train_df = index_df[index_df["ID"].isin(train_ids)].copy()
    val_df = index_df[index_df["ID"].isin(val_ids)].copy()

    return train_df, val_df

def age_to_bin(age):
    if pd.isna(age):
        return None
    age = int(age)
    if age <= 24:
        return "15_24"
    elif age <= 34:
        return "25_34"
    elif age <= 44:
        return "35_44"
    elif age <= 54:
        return "45_54"
    else:
        return "55_64"

def compute_sample_weights(train_df):
    id_counts = Counter(train_df["ID"])
    age_bin_counts = Counter(train_df["age_bin"])

    weights = []
    for _, row in train_df.iterrows():
        pid = row["ID"]
        age_bin = row["age_bin"]

        w_id = 1.0 / id_counts[pid]
        w_age = 1.0 / age_bin_counts[age_bin]

        weights.append(w_id * w_age)

    return weights


def load_or_create_index(root_dir, pkl_path):

    if os.path.exists(pkl_path):
        print("Loading index from PKL...")
        df = pd.read_pickle(pkl_path)
        return df

    print("Index not found. Building...")
    df = build_eeg_index(root_dir)

    print("Saving index...")
    df.to_pickle(pkl_path)

    return df