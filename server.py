import os
import sys

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from ModelDefs.AttAgeModel import AgeModel

app = FastAPI()
model = AgeModel(pretrained_dir="Modeling_ContrastiveChannelWise/ContrastiveModel_chkpoints",
                 hyperparam_config="FindOptimalProbForAugmentation/contrastive_hyperparams.json",
                 unlock_encoders=True,
                 config_file='config.json')
model.load_state_dict(torch.load("Modeling_ContrastiveChannelWise/ContrastiveModel_chkpoints/FullModel_Epoch22.pth", map_location="cuda:0"))
model.to("cuda:0")
model.eval()

class EEGInput(BaseModel):
    data: list  # expected to be a flat list of shape [512*14]

@app.post("/predict")
def predict_age(input_data: EEGInput):

    arr = np.array(input_data.data).reshape(1, 512, 14)
    x = torch.tensor(arr, dtype=torch.float32, device="cuda:0")

    with torch.no_grad():
        pred, attn_weights, segment_lengths = model(x)
    print("On server: ")
    #print(pred, attn_weights, segment_lengths)
    # Build feature labels and channel mapping
    feature_labels = []
    segment_to_channel = []
    for ch_idx, (ch_name, seg_count) in enumerate(zip(model.eeg_columns, segment_lengths)):
        feature_labels.extend([f"{ch_name}.L{i+1}" for i in range(seg_count)])
        segment_to_channel.extend([ch_idx] * seg_count)

    # Convert attention matrix to DataFrame with channel-level indices
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()  # shape: [S, S]

    segment_to_channel = np.array(segment_to_channel)  # shape: [S]
    n_channels = len(model.eeg_columns)

    channel_attn = np.zeros((n_channels, n_channels), dtype=np.float32)

    for i in range(n_channels):
        row_mask = segment_to_channel == i
        for j in range(n_channels):
            col_mask = segment_to_channel == j
            block = attn_np[np.ix_(row_mask, col_mask)]
            channel_attn[i, j] = block.mean() if block.size > 0 else 0.0
    return {
        "predicted_age": float(pred.item()),
        "feature_attention": attn_np.tolist(),         # [segments, segments]
        "channel_attention": channel_attn.tolist(),     # [14, 14]
        "feature_labels": feature_labels,
        "channel_labels": model.eeg_columns
    }


