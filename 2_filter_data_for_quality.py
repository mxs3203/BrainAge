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
MARKER_GROUPS = config["marker_groups"]
DEBUG = config["DEBUG"]
EEG_COLUMNS = config["EEG_columns"]
QUALITY = config["Quality"]
# Number of rows = 256samples/second×0.2seconds=51.2rows
NUMBER_OF_ROWS_DISCARDED = config["number_of_rows_discarded"]

EEG_COLUMNS.append(QUALITY)
if DEBUG:
    data_file = "results/BA7/0_total_data.csv"
    quality_file = "results/BA7/1_quality_filter.csv"
    output_file = "results/BA7/2_filtered_data.csv"
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]
else:
    data_file = sys.argv[1]
    quality_file = sys.argv[2]
    output_file = sys.argv[3]
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]

data = pd.read_csv(data_file)
quality = pd.read_csv(quality_file)
os.makedirs(output_folder, exist_ok=True)
filtered_stimuli = quality[quality["Filter"] != 'PASS']
print("The following stimuli were filtered: ", filtered_stimuli.Column.values)
quality = quality[quality["Filter"] == 'PASS']
good_stimuli = quality.Column.values
data = data[data['task_name'].isin(good_stimuli)]
data.to_csv("{}".format(output_file), index=False)
