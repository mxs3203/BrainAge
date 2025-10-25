import json
import os
import pickle
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

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
    output_file = "results/BA15/4_fft_data.pck"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
fft_results = {}
with open(data_file, "rb") as f:
    data = pickle.load(f)
fft_results = compute_fft_bands(data, config, fs=256)
with open(output_file, 'wb') as file:
    pickle.dump(fft_results, file)
    file.close()


