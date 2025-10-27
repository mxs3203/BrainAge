import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
import tqdm

with open("config.json", "r") as f:
    config = json.load(f)

'''
    Reading Configuration
'''
STIMULI = config["how_to_normalize"]
DEBUG = config["DEBUG"]
EEG_COLUMNS = config["EEG_columns"]
WINDOW_SIZE = config['window_size']
STRIDE = config['stride']
META_DATA_PATH = config["meta_data_path"]
OUTPUT_DIR = "results/EEG_LSTM_windows"

if DEBUG:
    data_file = "results/BA15/3_normalized_dictionary_of_stimuli.pck"
    output_file = "results/BA15/5_lstm_ready_flag.txt"
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]

meta_data_df = pd.read_excel(META_DATA_PATH, skiprows=1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(data_file, "rb") as f:
    data = pickle.load(f)
ID = data_file.split("/")[1]
meta_data_row = meta_data_df[meta_data_df['ID'] == ID]

example_key = next(iter(data))
example_df = next((df for df in data[example_key] if not df.empty), None)
if example_df is None:
    raise ValueError("No valid EEG segments found in the dictionary.")
eeg_cols = [c for c in example_df.columns if c.startswith("EEG.")]

total_windows = 0
for stim_name, seg_list in data.items():
    for df in seg_list:
        if df.empty:
            continue

        age = meta_data_row["Age"].values[0]

        eeg_data = df[eeg_cols].values
        n_samples, n_channels = eeg_data.shape

        # Sliding window
        for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            X_window = eeg_data[start:end]

            sample = {
                "X": X_window.astype(np.float32),
                "stimulus": stim_name,
                "ID": ID,
                "Age": float(age)
            }

            filename = f"ID_{ID}_win{total_windows}.pkl"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "wb") as f:
                pickle.dump(sample, f)
                f.close()

            total_windows += 1

# --- Write completion flag for Snakemake ---
with open(output_file, "w") as f:
    f.write(f"ID: {ID}\n")
    f.write(f"Total windows: {total_windows}\n")
    f.write(f"Output dir: {OUTPUT_DIR}\n")

