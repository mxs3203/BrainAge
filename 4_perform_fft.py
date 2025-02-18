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
    data_file = "results/BA15/3_normalized_data.csv"
    output_file = "results/BA15/4_fft_data.pck"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
fft_results = {}
df_dict = {}
data = pd.read_csv(data_file)
# for each stimuli
for stimuli in tqdm.tqdm(STIMULI):
    heatmap_data = {}
    # isolate the stimuli
    signal_df = data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS]
    if np.shape(signal_df)[0] >= 1:
        # for every column
        for channel in EEG_COLUMNS:
            channel_results = {}
            # for every FFT bands defined in config
            for band, (low, high) in FFT_BANDS.items():
                # isolate the values for each channel
                signal = signal_df[channel].values
                # perform FFT and extract frequncies
                fft_complex = np.fft.fft(signal)
                fft_magnitude = np.abs(fft_complex)
                freq = np.fft.fftfreq(len(signal), d=1/256)  # 256hz iz sampling rate
                # we are only interested in specific band
                band_mask = (freq >= low) & (freq < high)
                if np.any(band_mask):  # if they exist compute the mean and store it
                    channel_results[band] = np.mean(fft_magnitude[band_mask])
                else:
                    channel_results[band] = np.nan
            # store the data for this specific channel
            heatmap_data[channel] = channel_results
    else:
        print("No data for {}".format(stimuli))
    # this is EEG_columns by wavebands dataframe. Ideal for visualization
    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient='index')
    heatmap_df.index.name = "EEG Channel"
    heatmap_df = heatmap_df.reset_index()
    # add it to fina ldictionary by the name of stimuli
    df_dict[stimuli['input']] = heatmap_df

with open(output_file, 'wb') as file:
    pickle.dump(df_dict, file)
    file.close()


