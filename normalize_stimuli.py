import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm

from normalization_methods import z_score_normalization, basic_normalization, ratio_normalization, \
    detrend_normalization, IQR_normalization

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

if DEBUG:
    data_file = "results/BA1/0_total_data.csv"
    output_file = "results/BA1/1_normalized_data.csv"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
'''
    Main functionality
'''
data = pd.read_csv(data_file)
for stimuli in tqdm.tqdm(STIMULI):
    total_baseline = pd.DataFrame()
    # some stimulis can be normalized with more baselines
    for b in stimuli['normalizeWith']:
        tmp_baseline = data.loc[data['task_name'] == b, EEG_COLUMNS]
        # create combined DF with all baselines
        total_baseline = pd.concat([total_baseline, tmp_baseline])
    # isolate stimuli data
    signal_df = data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS]

    # perform the normalization
    if NORMALIZATION == "z-score":
        normalized_df = z_score_normalization(signal_df=signal_df, baseline_df=total_baseline)
        data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS] = normalized_df
    elif NORMALIZATION == "basic":
        normalized_df = basic_normalization(signal_df=signal_df, baseline_df=total_baseline)
        data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS] = normalized_df
    elif NORMALIZATION == "ratio":
        normalized_df = ratio_normalization(signal_df=signal_df, baseline_df=total_baseline)
        data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS] = normalized_df
    elif NORMALIZATION == "detrend":
        normalized_df = detrend_normalization(signal_df=signal_df)
        data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS] = normalized_df
    elif NORMALIZATION == "iqr":
        normalized_df = IQR_normalization(signal_df=signal_df, baseline_df=total_baseline)
        data.loc[data['task_name'] == stimuli['input'], EEG_COLUMNS] = normalized_df
    else:
        print("Unknown normalization method...")
        break

data.to_csv("{}".format(output_file), index=False)