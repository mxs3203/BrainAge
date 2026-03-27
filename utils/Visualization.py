import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import umap



def visualize_umap_age_and_id_structure(
    Z,
    age,
    ids,
    epoch,
    title_prefix,
    n_points=2000,
    max_ids_to_draw=20,
    random_state=42
):

    # -----------------------------
    # Convert inputs
    # -----------------------------
    if hasattr(Z, "detach"):
        Z = Z.detach().cpu().numpy()
    else:
        Z = np.asarray(Z)

    if hasattr(age, "detach"):
        age = age.detach().cpu().numpy()
    else:
        age = np.asarray(age)

    ids = np.asarray(ids)

    # -----------------------------
    # Optional subsampling
    # -----------------------------
    rng = np.random.default_rng(random_state)
    if Z.shape[0] > n_points:
        idx = rng.choice(Z.shape[0], n_points, replace=False)
        Z = Z[idx]
        age = age[idx]
        ids = ids[idx]

    # -----------------------------
    # UMAP
    # -----------------------------
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state
    )
    Z_2d = reducer.fit_transform(Z)

    df_plot = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "age": age,
        "id": ids
    })

    # IDs with at least 2 points
    id_counts = df_plot["id"].value_counts()
    valid_ids = id_counts[id_counts >= 2].index.tolist()

    # If too many IDs, draw only subset in panel B
    if len(valid_ids) > max_ids_to_draw:
        draw_ids = rng.choice(valid_ids, size=max_ids_to_draw, replace=False)
    else:
        draw_ids = valid_ids

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    # =====================================================
    # Panel A: Age-colored UMAP
    # =====================================================
    ax = axes[0]

    scatter = ax.scatter(
        df_plot["x"],
        df_plot["y"],
        c=df_plot["age"],
        cmap="bwr",
        s=18,
        alpha=0.85,
        edgecolors="none"
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Age")

    ax.set_title("A. UMAP colored by age", loc="left", fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # =====================================================
    # Panel B: Same-ID spread
    # =====================================================
    ax = axes[1]

    # Light background points
    ax.scatter(
        df_plot["x"],
        df_plot["y"],
        color="0.85",
        s=12,
        alpha=0.15,
        edgecolors="none",
        zorder=0
    )

    cmap = plt.get_cmap("tab20")
    within_distances = []

    for i, uid in enumerate(draw_ids):
        sub = df_plot[df_plot["id"] == uid].copy()
        if len(sub) < 2:
            continue

        cx = sub["x"].mean()
        cy = sub["y"].mean()

        color = cmap(i % 20)

        # Lines from centroid to points
        for _, row in sub.iterrows():
            ax.plot(
                [cx, row["x"]],
                [cy, row["y"]],
                color=color,
                alpha=0.15,
                linewidth=0.8,
                zorder=1
            )

            d = np.sqrt((row["x"] - cx) ** 2 + (row["y"] - cy) ** 2)
            within_distances.append(d)

        # Points for this ID
        ax.scatter(
            sub["x"],
            sub["y"],
            s=20,
            alpha=0.85,
            color=color,
            edgecolors="none",
            zorder=2
        )

        # Centroid with same fill color + red border
        ax.scatter(
            [cx], [cy],
            s=85,
            color=color,
            edgecolors="red",
            linewidths=1.6,
            zorder=4
        )

    ax.set_title("B. Same-ID spread in UMAP space", loc="left", fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Summary text
    if len(within_distances) > 0:
        mean_within = float(np.mean(within_distances))
        median_within = float(np.median(within_distances))
        ax.text(
            0.02, 0.98,
            f"Mean centroid distance: {mean_within:.2f}\n"
            f"Median centroid distance: {median_within:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="black",
                alpha=0.85
            )
        )

    fig.suptitle(f"{title_prefix} | epoch={epoch}", fontsize=14, fontweight="bold")

    plt.savefig(
        f"../MainFigures/{title_prefix}_{epoch}_age_id_structure.pdf",
        bbox_inches="tight"
    )

    return fig

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