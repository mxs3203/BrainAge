import json
import os
import pickle
import sys
import pandas as pd
import tqdm

from utils.normalization_methods import z_score_normalization, basic_normalization, ratio_normalization, \
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
    data_file = "results/BA1/2_non_normalized_dictionary_of_stimuli.pck"
    output_file = "results/BA1/3_normalized_dictionary_of_stimuli.pck"
    output_folder = "/".join(output_file.split("/")[:2])
else:
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    output_folder = "/".join(output_file.split("/")[:2])


norm_func_map = {
        "z-score": z_score_normalization,
        "basic": basic_normalization,
        "ratio": ratio_normalization,
        "IQR": IQR_normalization,
        "detrend": detrend_normalization
}
norm_func = norm_func_map[NORMALIZATION.lower()]

os.makedirs(output_folder, exist_ok=True)
'''
    Main functionality
'''
with open(data_file, "rb") as f:
    data_dict = pickle.load(f)

# this will be the output
normalized_segments = {}
# for all stimuli that needs to be normalized
for entry in tqdm.tqdm(config["how_to_normalize"]):
    task_name = entry["input"]
    baselines = entry["normalizeWith"]

    if task_name not in data_dict:
        print("{} is not in dictionary... Probably filtered by quality. skipping...". format(task_name))
        continue

    baseline_dfs = []
    # merge baselines
    for base in baselines:
        if base in data_dict and len(data_dict[base]) > 0:
            # Take all segments and concatenate
            baseline_dfs.append(pd.concat(data_dict[base], ignore_index=True))
    baseline_df = pd.concat(baseline_dfs, ignore_index=True)

    # create an empty list which will be used to store normalized DFs
    normalized_segments[task_name] = []

    for seg_df in data_dict[task_name]:
        norm_df = norm_func(seg_df.filter(like="EEG."), baseline_df.filter(like="EEG."))
        normalized_segments[task_name].append(norm_df)

with open(output_file, "wb") as f:
    pickle.dump(normalized_segments, f)