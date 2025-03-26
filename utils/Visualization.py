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

