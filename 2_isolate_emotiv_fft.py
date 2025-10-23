import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

with open("config.json", "r") as f:
    config = json.load(f)

'''
    Reading Configuration
'''
DEBUG = config["DEBUG"]
FFT_COLUMNS = config["pow_columns"]
FFT_QUALITY_THRS = config["filter_for_extract_emotiv_fft"]
FFT_COLUMNS += ["task_name", "ID"]
FFT_COLUMNS += config['channel_quality_columns']
threshold = 2

if DEBUG:
    data_file = "results/BA1/0_total_data.csv"
    output_file = "results/BA1/2_emotiv_fft_values.csv"
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]

os.makedirs(output_folder, exist_ok=True)
'''
    Main functionality
'''
data = pd.read_csv(data_file)
fft_data = data[FFT_COLUMNS]
fft_data = fft_data.dropna()
mask = (data[config['channel_quality_columns']] > FFT_QUALITY_THRS).all(axis=1)
filtered_fft_data = fft_data[mask.loc[fft_data.index]]
fft_data.to_csv("{}".format(output_file), index=False)