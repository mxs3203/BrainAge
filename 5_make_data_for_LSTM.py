import json
import os
import pickle
import random
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

with open("config.json", "r") as f:
    config = json.load(f)

'''
    Reading Configuration
'''
STIMULI = config["how_to_normalize"]
DEBUG = config["DEBUG"]
EEG_COLUMNS = config["EEG_columns"]
QUALITY = config["Quality"]
NORMALIZATION = config["normalization"]
FFT_BANDS = config["fft_bands"]
WINDOW_SIZE = 100
QUALITY_THRESHOLD = config["filter_pass_threshold"]['mean']
META_DATA_PATH = config["meta_data_path"]

if DEBUG:
    data_file = "results/BA15/3_normalized_data.csv"
    output_file = "results/BA15/7_LSTM_data.pck"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])

meta_data_df = pd.read_excel(META_DATA_PATH,skiprows=1)
EEG_COLUMNS.append(QUALITY)
os.makedirs(output_folder, exist_ok=True)
fft_results = {}
df_dict = {}
data = pd.read_csv(data_file)
ID = data['ID'].unique()[0]
meta_data_df = meta_data_df[meta_data_df['ID'] == ID]
# for each stimuli
windows = []
for stimuli in tqdm.tqdm(STIMULI):
    signal_df = data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS]
    for start in range(0, len(signal_df) - WINDOW_SIZE + 1, 1):
        window = signal_df.iloc[start:start + WINDOW_SIZE]
        if (window[QUALITY] > QUALITY_THRESHOLD).all():
            eeg_data = window[EEG_COLUMNS[:-1]].values  # shape: (window_size, 14)
            windows.append({
                "ID": ID,
                "stimulus": stimuli['input'],
                "data": eeg_data,
                "Age": meta_data_df['Age'].values[0] + random.uniform(-1, 1),
            })
pickle.dump(windows, open(output_file, "wb"))