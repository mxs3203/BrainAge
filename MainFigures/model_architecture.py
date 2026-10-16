"""
model_architecture.py — BrainAge model architecture diagram.

Panel A: EEGConvGRUEncoder detail (one channel, pretrained contrastively)
Panel B: Full AgeModel flow (14 encoders → attention → FC → age)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Uniform style ──────────────────────────────────────────────────────────────
BOX_FC  = "#F0F0F0"   # light gray fill
BOX_EC  = "#999999"   # gray border
BOX_LW  = 1.6
RADIUS  = "round,pad=0.018"
FS      = 10.5        # main text font size
FS_SUB  = 8.5         # sub-label font size
ARROW_C = "#666666"
ARROW_W = 1.8
GAP     = 0.013       # vertical gap between box centres

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.weight": "bold",
    "text.color": "#111111",
})


def box(ax, cx, cy, w, h, text, sub=""):
    """Draw a light-gray rounded box at (cx, cy) with optional sub-label."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=RADIUS,
        facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=BOX_LW, zorder=3,
    ))
    if sub:
        ax.text(cx, cy + h * 0.13, text,
                ha="center", va="center", fontsize=FS,
                fontweight="bold", color="#111111", zorder=4)
        ax.text(cx, cy - h * 0.20, sub,
                ha="center", va="center", fontsize=FS_SUB,
                fontweight="normal", color="#555555", zorder=4)
    else:
        ax.text(cx, cy, text,
                ha="center", va="center", fontsize=FS,
                fontweight="bold", color="#111111", zorder=4)


def arrow(ax, cx, y_top, y_bot):
    """Draw a centred downward arrow from y_top to y_bot."""
    ax.annotate("", xy=(cx, y_bot), xytext=(cx, y_top),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                lw=ARROW_W, mutation_scale=12), zorder=2)


def panel_axis(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 11), facecolor="white")
fig.text(0.5, 0.985,
         "BrainAge Model Architecture",
         ha="center", va="top", fontsize=15, fontweight="bold")
fig.text(0.5, 0.965,
         "14 per-channel contrastive encoders  →  cross-channel attention  →  age regression",
         ha="center", va="top", fontsize=10, color="#444444")

# ── Panel A — encoder detail ───────────────────────────────────────────────────
ax_a = panel_axis(fig, [0.03, 0.04, 0.42, 0.91])
ax_a.text(0.5, 0.985, "A.  EEGConvGRUEncoder  (per channel, pretrained)",
          ha="center", va="top", fontsize=11, fontweight="bold")

cx = 0.50
W  = 0.80
BH = 0.082
step = BH + GAP

# Block definitions: (y_centre, main text, sub-label)
enc_blocks = [
    (0.900, "EEG channel input",
             "[B, 512, 1]"),
    (0.900 - step,   "Conv1d  →  BatchNorm  →  Activation  →  Dropout",
             "1 → C channels,  kernel k, stride s  →  [B, C, L′]"),
    (0.900 - 2*step, "Permute  →  GRU / LSTM",
             "[B, L′, C]  →  hidden size H,  num_layers  →  last h_n  [B, H]"),
    (0.900 - 3*step, "Projection head:  Linear(H → D)  +  L2 Normalise",
             "contrastive embedding  [B, D]"),
    (0.900 - 4*step, "Channel Projector:  Linear(D → 16)  +  LayerNorm",
             "[B, 16]  —  per-channel feature fed into AgeModel"),
]

for i, (y, txt, sub) in enumerate(enc_blocks):
    box(ax_a, cx, y, W, BH, txt, sub)
    if i < len(enc_blocks) - 1:
        arrow(ax_a, cx, y - BH/2, enc_blocks[i+1][0] + BH/2)

# ── Panel B — full model ───────────────────────────────────────────────────────
ax_b = panel_axis(fig, [0.55, 0.04, 0.43, 0.91])
ax_b.text(0.5, 0.985, "B.  AgeModel  (full pipeline)",
          ha="center", va="top", fontsize=11, fontweight="bold")

cx = 0.50
W  = 0.88
BH = 0.070
step = BH + GAP

mdl_blocks = [
    (0.900, "EEG Input",
             "[B × 512 × 14]   (14 channels, 512 time steps)"),
    (0.900 - step,   "EEGConvGRUEncoder  ×  14  (pretrained)",
             "one independent encoder per channel  →  embedding [B, D_ch]"),
    (0.900 - 2*step, "Channel Projectors  ×  14",
             "Linear(D_ch → 16)  +  LayerNorm  →  [B, 16] per channel"),
    (0.900 - 3*step, "Stack channel features",
             "[B, 14, 16]"),
    (0.900 - 4*step, "Multi-Head Self-Attention  (across channels)",
             "embed_dim = 16,  heads = 2   →  [B, 14, 16]  +  weights [B, 2, 14, 14]"),
    (0.900 - 5*step, "Flatten",
             "[B, 14 × 16]  =  [B, 224]"),
    (0.900 - 6*step, "FC1:  Linear(224 → 512)  +  Dropout(0.1)  +  SiLU",
             ""),
    (0.900 - 7*step, "FC2:  Linear(512 → 512)  +  SiLU",
             ""),
    (0.900 - 8*step, "Output:  Linear(512 → 1)",
             ""),
    (0.900 - 9*step, "Predicted Brain Age  (years)",
             "[B]  —  regression output"),
]

for i, (y, txt, sub) in enumerate(mdl_blocks):
    box(ax_b, cx, y, W, BH, txt, sub)
    if i < len(mdl_blocks) - 1:
        arrow(ax_b, cx, y - BH/2, mdl_blocks[i+1][0] + BH/2)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"model_architecture.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  → {p}")
plt.close(fig)
print("Done.")
