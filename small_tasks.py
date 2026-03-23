import os

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import glob

# Path pattern: BA*/4_EEG_features.csv
# pattern = "results/BA*/2_emotiv_fft_values.csv"
#
# dfs = []
#
# for file_path in glob.glob(pattern):
#     # Extract folder name (BA1, BA2, ...)
#     ba_id = os.path.basename(os.path.dirname(file_path))
#     print(ba_id)
#     # Read CSV
#     df = pd.read_csv(file_path)
#
#     # Add ID column
#     df["ID"] = ba_id
#
#     dfs.append(df)
#
# # Merge all into one DataFrame
# merged_df = pd.concat(dfs, ignore_index=True)
#
# # Optional: save result
# merged_df.to_csv("EEG_FFT_merged.csv", index=False)
#
# print(f"Merged {len(dfs)} files")
# print(merged_df.head())


df = pd.read_csv("EEG_FFT_merged.csv")
pow_cols = [c for c in df.columns if c.startswith("POW.")]

# 2. Melt to long format
long = df.melt(
    id_vars=["ID", "task_name"],
    value_vars=pow_cols,
    var_name="feature",
    value_name="value"
)

# 3. Split feature name into components
# feature format: POW.AF4.Alpha
long[["prefix", "channel", "band"]] = long["feature"].str.split(".", expand=True)

# 4. Pivot so bands become columns
final_df = (
    long
    .pivot_table(
        index=["ID", "task_name", "channel"],
        columns="band",
        values="value"
    )
    .reset_index()
)

# Optional: remove column index name
final_df.columns.name = None
final_df.to_csv("FFT_NMF_Matrix.csv")