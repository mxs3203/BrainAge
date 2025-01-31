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
    data_file = "data/BA1/data.csv"
    markers_file = "data/BA1/marker.csv"
    output_file = "results/BA1/total_data.csv"
    output_file_stat = "results/BA1/mean_sd.csv"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    markers_file = sys.argv[2]
    output_file = sys.argv[3]
    output_file_stat = sys.argv[4]
    output_folder = "/".join(output_file.split("/")[:2])


os.makedirs(output_folder, exist_ok=True)
'''
    Main functionality
'''
data = pd.read_csv(data_file, skiprows=1)
markers = pd.read_csv(markers_file)
markers = markers[markers[markers['type'] == "eyesopen"].index.values[0]:]
markers = markers[markers['type'] != "active_period"]

all_markers = dict(zip(markers['marker_id'], markers['type']))

total_data = pd.DataFrame()
mean_sd_values = []
for marker in all_markers.keys():
    start_i = data[data["MarkerIndex"] == marker].index
    end_i = data[data["MarkerIndex"] == -marker].index
    if not start_i.empty and not end_i.empty:
        start = start_i.values[0]
        end = end_i.values[0]
    else:
        print(f"Marker {marker} not found in data.")
        break


    stimuli = data[(start+NUMBER_OF_ROWS_DISCARDED):(end-NUMBER_OF_ROWS_DISCARDED)][EEG_COLUMNS]
    stimuli['task_name'] = all_markers[marker]
    mean_sd_values.append({'Column': all_markers[marker], 'Mean': stimuli[QUALITY].mean(), 'StdDev': stimuli[QUALITY].std(), 'NumberOfRows': np.shape(stimuli)[0]})
    total_data = pd.concat([total_data, stimuli], ignore_index=True)

total_data.to_csv("{}".format(output_file), index=False)
mean_sd_df = pd.DataFrame(mean_sd_values)
mean_sd_df.to_csv("{}".format(output_file_stat), index=False)