import json
import os
import sys

import pandas as pd

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
PASS_THRESHOLD_mu = config["filter_pass_threshold"]['mean']
PASS_THRESHOLD_std = config["filter_pass_threshold"]['std']

if DEBUG:
    data_file = "results/BA1/0_mean_sd.csv"
    output_file = "results/BA1/1_quality_filter.csv"
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]

os.makedirs(output_folder, exist_ok=True)

quality_df = pd.read_csv(data_file)
# Assign 'PASS' or 'FAIL' based on Mean value
quality_df["Filter"] = quality_df.apply(
    lambda row: "PASS" if (row["Mean"] >= PASS_THRESHOLD_mu and row["StdDev"] <= PASS_THRESHOLD_std) else "FAIL",
    axis=1
)
quality_df['ID'] = ID
quality_df.to_csv(output_file, index=False)
