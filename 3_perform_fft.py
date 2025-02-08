import json
import os
import pickle
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

if DEBUG:
    data_file = "results/BA1/2_normalized_data.csv"
    output_file = "results/BA1/3_fft_data.pck"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
fft_results = {}
df_dict = {}
data = pd.read_csv(data_file)
for stimuli in tqdm.tqdm(STIMULI):
    heatmap_data = {}
    signal_df = data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS]
    for channel in EEG_COLUMNS:
        channel_results = {}
        for band, (low, high) in FFT_BANDS.items():

            signal = signal_df[channel].values
            fft_complex = np.fft.fft(signal)
            fft_magnitude = np.abs(fft_complex)
            freq = np.fft.fftfreq(len(signal), d=1/256)  # 256hz iz sampling rate

            band_mask = (freq >= low) & (freq < high)
            if np.any(band_mask):  # Ensure there are frequency bins in this band.
                channel_results[band] = np.mean(fft_magnitude[band_mask])
            else:
                channel_results[band] = np.nan
        heatmap_data[channel] = channel_results
    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient='index')
    heatmap_df.index.name = "EEG Channel"
    heatmap_df = heatmap_df.reset_index()
    df_dict[stimuli['input']] = heatmap_df

with open(output_file, 'wb') as file:
    pickle.dump(df_dict, file)
    file.close()


