import json
import os
import pickle
import sys
import seaborn as sns
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
QUALITY = config["Quality"]
NORMALIZATION = config["normalization"]
STIMULI_GROUPS = config["marker_groups"]
BRAIN_SCORE_PARAMS = config["brain_score_fft"]

if DEBUG:
    data_file = "results/BA1/3_fft_data.pck"
    output_file = "results/BA1/5_brain_score_fft.csv"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])

group_scores = {}
os.makedirs(output_folder, exist_ok=True)
with open(data_file, 'rb') as file:
    data = pickle.load(file)

group_scores = {}
for group, stimuli_list in STIMULI_GROUPS.items():
    group_data = pd.DataFrame()
    for stimuli in stimuli_list:
        tmp = data[stimuli][BRAIN_SCORE_PARAMS]
        group_data = pd.concat([group_data, tmp])
    group_scores[group] = group_data.mean(axis=0).sum()

df_scores = pd.DataFrame([group_scores])
df_scores["BrainScore_fft"] = np.log10(df_scores.sum(axis=1))
df_scores.to_csv(output_file, index=False)