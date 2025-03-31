import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_heatmaps(epoch, attn_weights, output_var, channel_names=[
        "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7",
        "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
    ], samples_to_plot=4):
    batch_size, num_channels,_ = attn_weights.shape
    attn_weights = attn_weights.detach().cpu().numpy()
    output_var = output_var.detach().cpu().numpy()

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]

    for i in range(min(samples_to_plot, batch_size)):
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn_weights[i], xticklabels=channel_names, yticklabels=channel_names,
                    cmap="viridis", annot=False, square=True, cbar=True)
        plt.title(f"Attention Heatmap - Sample {i} with Age {output_var[i]}")
        plt.xlabel("Attended Channel")
        plt.ylabel("Query Channel")
        plt.tight_layout()
        plt.savefig(f"Plots/AttentionHeatmap-{i}-Epoch-{epoch}.pdf")


# Helps interpret which brain regions (electrodes) are most influential
def plot_channel_importance(attn_weights, channel_names=[
        "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7",
        "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"
    ], title="Channel Attention Importance"):
    # Average across batch and heads
    mean_attn = attn_weights.mean(dim=0)  # [14, 14]
    channel_scores = mean_attn.mean(dim=0)  # column-wise mean

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(len(channel_scores))]

    # Plot barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=channel_names, y=channel_scores.cpu().numpy())
    plt.title(title)
    plt.ylabel("Average Attention Received")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_channel_attention_scores(attn_weights, channel_names):
    """
    Compute average attention received per channel.
    Returns a dictionary mapping channel → attention score.
    """
    mean_attn = attn_weights.mean(dim=0)  # [num_channels, num_channels]
    channel_scores = mean_attn.mean(dim=0)  # column-wise mean: how much each channel is attended to

    return dict(zip(channel_names, channel_scores.cpu().numpy()))

def save_attention_scores_to_csv(attn_weights, channel_names, output_path="channel_attention_scores.csv"):
    scores = get_channel_attention_scores(attn_weights, channel_names)
    df = pd.DataFrame(list(scores.items()), columns=["Channel", "AttentionScore"])
    df.to_csv(output_path, index=False)
    print(f"📁 Attention scores saved to: {output_path}")

'''
_, attn_weights = model(x)  # [batch, 14, 14]

save_attention_scores_to_csv(attn_weights, output_path="attention_report.csv")
🧠

region_map = {
    "AF3": "Frontal", "AF4": "Frontal", "F3": "Frontal", "F4": "Frontal", "F7": "Frontal", "F8": "Frontal",
    "FC5": "Frontal", "FC6": "Frontal", "T7": "Temporal", "T8": "Temporal", "P7": "Parietal", "P8": "Parietal",
    "O1": "Occipital", "O2": "Occipital"
}

df["Region"] = df["Channel"].map(region_map)
df.to_csv("attention_report_with_regions.csv", index=False)

# Your channel attention scores:
channel_scores = {
    "AF3": 0.1, "F7": 0.05, "F3": 0.08, "FC5": 0.07, "T7": 0.03, "P7": 0.06, "O1": 0.02,
    "O2": 0.04, "P8": 0.05, "T8": 0.04, "FC6": 0.08, "F4": 0.09, "F8": 0.06, "AF4": 0.1
}

# Create MNE montage and info
ch_names = list(channel_scores.keys())
values = np.array(list(channel_scores.values()))

# Assume all EEG
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
info.set_montage(montage)

# Create fake Evoked object just to use topomap plotting
fake_data = np.expand_dims(values, axis=1)  # shape (n_channels, 1)
evoked = mne.EvokedArray(fake_data, info)
evoked.set_montage(montage)

# Plot topomap
evoked.plot_topomap(times=0, scalings=1, time_format='', cmap='viridis', size=3.0)
plt.suptitle('Average Channel Attention (Topomap)', fontsize=14)
plt.show()

https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html
'''