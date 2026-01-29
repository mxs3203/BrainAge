import os

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import glob

# Path pattern: BA*/4_EEG_features.csv
pattern = "results/BA*/4_EEG_features.csv"

dfs = []

for file_path in glob.glob(pattern):
    # Extract folder name (BA1, BA2, ...)
    ba_id = os.path.basename(os.path.dirname(file_path))

    # Read CSV
    df = pd.read_csv(file_path)

    # Add ID column
    df["ID"] = ba_id

    dfs.append(df)

# Merge all into one DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Optional: save result
merged_df.to_csv("EEG_features_merged.csv", index=False)

print(f"Merged {len(dfs)} files")
print(merged_df.head())
