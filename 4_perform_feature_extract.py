import json
import os
import pickle
import sys
import pandas as pd
import tqdm

from utils.SignalFeatureExtractor import EEGFeatureExtractor
from utils.fft import compute_fft_bands

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
    data_file = "results/BA15/2_non_normalized_dictionary_of_stimuli.pck"
    normalized_data_file = "results/BA15/3_normalized_dictionary_of_stimuli.pck"
    output_file = "results/BA15/4_fft_data.pck"
    output_feature_file = "results/BA15/4_features.csv"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    normalized_data_file = sys.argv[2]
    output_file = sys.argv[3]
    output_feature_file = sys.argv[4]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
fft_results = {}
with open(data_file, "rb") as f:
    data = pickle.load(f)
fft_results = compute_fft_bands(data, config, fs=256)
with open(output_file, 'wb') as file:
    pickle.dump(fft_results, file)
    file.close()

################################## EEG FEATURES ###########################################

with open(normalized_data_file, "rb") as f:
    data = pickle.load(f)
total_df = pd.DataFrame()
for task, seg_list in tqdm.tqdm(data.items()):
    print(task)
    for seg_df in seg_list:
        extractor = EEGFeatureExtractor(seg_df[EEG_COLUMNS])
        features_df = extractor.compute()
        features_df['Stimuli'] = task
        features_df['Channel'] = features_df.index
        total_df = pd.concat([total_df, features_df])

total_df.to_csv(output_feature_file)


