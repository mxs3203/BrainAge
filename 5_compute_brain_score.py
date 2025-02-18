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

if DEBUG:
    data_file = "results/BA1/2_normalized_data.csv"
    output_file = "results/BA1/4_brain_score.csv"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])

group_scores = {}
os.makedirs(output_folder, exist_ok=True)
data = pd.read_csv(data_file)
for group, stimuli_list in STIMULI_GROUPS.items():
    filtered_df = data[data['task_name'].isin(stimuli_list)]
    group_score = filtered_df[EEG_COLUMNS].abs().mean().mean()
    group_scores[group] = group_score
df_scores = pd.DataFrame([group_scores])
df_scores["BrainScore"] = df_scores.sum(axis=1)
df_scores.to_csv(output_file, index=False)