import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ModelDefs.AttAgeModel import AgeModel


DATA_DIR = Path("results/EEG_LSTM_windows")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_age_model(device):
    model = AgeModel(pretrained_dir="Modeling_ContrastiveChannelWise/ContrastiveModel_chkpoints",
                     hyperparam_config="FindOptimalProbForAugmentation/contrastive_hyperparams.json",
                     unlock_encoders=True,
                     config_file='config.json')
    state = torch.load(
        "Modeling_ContrastiveChannelWise/ContrastiveModel_chkpoints/FullModel_Epoch24.pth",
        map_location=device
    )
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_sample_fields(obj, file_path):
    file_id = obj["ID"]

    eeg_data = obj["X"]
    true_age = obj["Age"]      # adjust if needed
    stimuli_name = obj["stimulus"]  # adjust if needed

    eeg_data = np.asarray(eeg_data, dtype=np.float32)

    if eeg_data.shape == (1, 512, 14):
        eeg_data = eeg_data[0]
    elif eeg_data.shape == (14, 512):
        eeg_data = eeg_data.T
    elif eeg_data.shape != (512, 14):
        raise ValueError(f"Unexpected shape {eeg_data.shape} in {file_path}")

    return file_id, stimuli_name, true_age, eeg_data


def build_feature_labels(channel_names, segment_lengths):
    feature_labels = []
    for ch_name, seg_count in zip(channel_names, segment_lengths):
        feature_labels.extend([f"{ch_name}_L{i+1}" for i in range(seg_count)])
    return feature_labels


def predict_direct_with_latent(model, eeg_data, device):
    x = torch.tensor(eeg_data[None, ...], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred, attn_weights, segment_lengths = model(x)

    pred_age = float(pred.item())
    latent_vec = attn_weights.squeeze(0).detach().cpu().numpy()
    feature_labels = build_feature_labels(model.eeg_columns, segment_lengths)

    if len(feature_labels) != len(latent_vec):
        raise ValueError(
            f"Mismatch: {len(feature_labels)} labels vs {len(latent_vec)} latent features"
        )

    return pred_age, latent_vec, feature_labels


def build_csv_direct(data_dir, model, device):
    files = list(data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    writer = None
    header = None
    min_age = 19
    max_age = 63
    with open("brain_age_results.csv", "w", newline="", encoding="utf-8") as f:
        for file_path in tqdm(files, desc="Processing EEG windows"):
            obj = load_pickle_file(file_path)
            file_id, stimuli_name, true_age, eeg_data = extract_sample_fields(obj, file_path)

            pred_age, latent_vec, feature_labels = predict_direct_with_latent(model, eeg_data, device)

            latent_vec = np.diag(latent_vec)

            row = {
                "ID": file_id,
                "StimuliName": stimuli_name,
                "TrueAge": true_age,
                "PredictedAge": pred_age,
                "PredictedAgeScaled": pred_age * (max_age - min_age) + min_age
            }

            for col_name, value in zip(feature_labels, latent_vec):
                row[col_name] = float(value)

            if writer is None:
                header = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

            writer.writerow(row)


if __name__ == "__main__":
    # model = load_age_model(DEVICE)
    # build_csv_direct(DATA_DIR, model, DEVICE)
    data = pd.read_csv("brain_age_results.csv")
    df_median = (
        data
        .groupby(["ID", "StimuliName"], as_index=False)
        .median(numeric_only=True)
    )

    df_median.to_csv("median_per_ID_stimulus.csv", index=False)
    df = pd.read_csv("median_per_ID_stimulus.csv")
    mse = mean_squared_error(df["TrueAge"], df["PredictedAge"])

    print("MSE:", mse)

    eeg_cols = [c for c in df.columns if c.startswith("EEG.")]

    channels = set(c.split("_")[0] for c in eeg_cols)

    for ch in channels:
        ch_cols = [c for c in eeg_cols if c.startswith(ch)]
        mean = df[ch_cols].values.mean()
        std = df[ch_cols].values.std()

        df[ch_cols] = (df[ch_cols] - mean) / std

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    df[eeg_cols] = scaler.fit_transform(df[eeg_cols])
    df.to_csv("median_per_ID_stimuli_normalized.csv", index=False)