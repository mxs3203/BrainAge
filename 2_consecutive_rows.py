import json
import os
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils.row_extract import extract_quality_segments

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
OVERALL_Q_THRS = config['overal_q_thrs']
CHANNEL_Q_THRS = config['channel_q_thrs']

if DEBUG:
    data_file = "results/BA68/0_total_data.csv"
    output_file = "results/BA68/2_non_normalized_dictionary_of_stimuli.pck"
    output_report_file = "results/BA68/2_consecutive_rows_report.csv"
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_report_file = sys.argv[3]
    output_folder = "/".join(output_file.split("/")[:2])
    ID = output_folder.split("/")[1]

data = pd.read_csv(data_file)
os.makedirs(output_folder, exist_ok=True)
segments, report = extract_quality_segments(
    data,
    cq_overall_col='CQ.Overall',
    cq_prefix='CQ.',
    task_col='task_name',
    overall_thrs=OVERALL_Q_THRS,
    channel_thrs=CHANNEL_Q_THRS
)
with open(output_file, "wb") as f:
    pickle.dump(segments, f)

report.to_csv("{}".format(output_report_file), index=False)
