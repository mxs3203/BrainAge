'''
FIGURE 5 — Age model hyperparameter sweep
  Panel A: Frozen vs fine-tuned encoders (boxplot + strip)
  Panel B: Common dim vs val loss  + poly trend
  Panel C: FC1 dim vs val loss     + poly trend
  Panel D: FC2 dim vs val loss     + poly trend
  Panel E: Attention heads vs val loss + poly trend
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "MainFigures/F5/age_model_runs.csv"
PDF_PATH = "MainFigures/F5/F5.pdf"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["best_val_loss"]).copy()

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "axes.edgecolor":    "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
})

blue = "#123B7A"
red  = "#C62828"
gray = "#6E6E6E"

def style_axis(ax):
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

frozen_df   = df[~df["fine_tune"]].copy()
finetune_df = df[ df["fine_tune"]].copy()
rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
axes = axes.flatten()

# ── Panel A: Frozen vs Fine-tuned overview ─────────────────────────────────────
ax = axes[0]
groups = [frozen_df["best_val_loss"].values, finetune_df["best_val_loss"].values]
labels = ["Frozen\nencoders", "Fine-tuned\nencoders"]
colors = [blue, red]

bp = ax.boxplot(groups, patch_artist=True, widths=0.45, tick_labels=labels)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor("black")
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)
for elem in bp["whiskers"] + bp["caps"]:
    elem.set_color("black")

for i, (vals, color) in enumerate(zip(groups, colors), start=1):
    x_jitter = rng.normal(i, 0.06, size=len(vals))
    ax.scatter(x_jitter, vals, s=45, color=color, edgecolors="black",
               linewidths=0.6, zorder=4, alpha=0.9)

ax.set_title("A. Frozen vs fine-tuned encoders", loc="left", fontweight="bold")
ax.set_ylabel("Best validation loss (SmoothL1)")
ax.grid(True, axis="y", alpha=0.25)
ax.set_axisbelow(True)
style_axis(ax)

# ── Shared helper: scatter + group mean line ──────────────────────────────────
def scatter_panel(ax, col, title, xlabel, letter, use_log2=True):
    x_raw = df[col].values.astype(float)
    y     = df["best_val_loss"].values.astype(float)
    x     = np.log2(x_raw) if use_log2 else x_raw

    # scatter — all points, single colour, small jitter
    x_jitter = x + rng.normal(0, 0.06 if use_log2 else 0.07, size=len(x))
    ax.scatter(x_jitter, y, s=55, color=blue, edgecolors="black",
               linewidths=0.6, alpha=0.9, zorder=3)

    # mean val loss per unique x value, connected by a line
    means = df.groupby(col)["best_val_loss"].mean().reset_index().sort_values(col)
    x_means = np.log2(means[col].values) if use_log2 else means[col].values
    ax.plot(x_means, means["best_val_loss"].values, color=red, linewidth=2,
            linestyle="--", marker="o", markersize=6, zorder=4, label="group mean")

    # x ticks at actual values
    unique_vals = sorted(df[col].unique())
    ax.set_xticks(np.log2(unique_vals) if use_log2 else unique_vals)
    ax.set_xticklabels([str(int(v)) for v in unique_vals])

    ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best validation loss (SmoothL1)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

scatter_panel(axes[1], "commonDim", "Common dim vs validation loss",
              "Common dim (attention embed size)", "B")

# ── Panel C: FC1 + FC2 combined as "size of linear layers" ───────────────────
ax = axes[2]
stacked = pd.concat([
    df[["best_val_loss"]].assign(dim=df["fc1_dim"], layer="FC1"),
    df[["best_val_loss"]].assign(dim=df["fc2_dim"], layer="FC2"),
], ignore_index=True)

for layer, marker in [("FC1", "o"), ("FC2", "s")]:
    sub = stacked[stacked["layer"] == layer]
    x   = np.log2(sub["dim"].values.astype(float))
    x_jitter = x + rng.normal(0, 0.06, size=len(x))
    ax.scatter(x_jitter, sub["best_val_loss"].values,
               s=55, color=blue, edgecolors="black", linewidths=0.6,
               alpha=0.9, marker=marker, zorder=3, label=layer)

# single mean line across both fc1 and fc2 values
means = stacked.groupby("dim")["best_val_loss"].mean().reset_index().sort_values("dim")
x_means = np.log2(means["dim"].values.astype(float))
ax.plot(x_means, means["best_val_loss"].values, color=red, linewidth=2,
        linestyle="--", marker="o", markersize=6, zorder=4, label="group mean")

unique_vals = sorted(stacked["dim"].unique())
ax.set_xticks(np.log2(unique_vals))
ax.set_xticklabels([str(int(v)) for v in unique_vals])
ax.set_title("C. Linear layer size vs validation loss", loc="left", fontweight="bold")
ax.set_xlabel("Layer size (FC1 = circle, FC2 = square)")
ax.set_ylabel("Best validation loss (SmoothL1)")
ax.grid(True, axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8)
style_axis(ax)

scatter_panel(axes[3], "att_heads", "Attention heads vs validation loss",
              "Number of attention heads", "D", use_log2=True)

fig.suptitle(
    "Age model hyperparameter sweep — frozen vs fine-tuned encoders",
    fontsize=14, fontweight="bold"
)

fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {PDF_PATH}")
