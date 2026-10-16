"""Build the Methods search-space table and Supplementary Table 1 as a .docx."""
import os
import sys

import torch
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "strengthen", "final_encoders")
OUT = os.path.join(HERE, "results", "Methods_tables.docx")

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# The searched space, as written in channel_encoder_search_arch.py
# (suggest_scalars + suggest_arch). Ranges are the union actually sampled,
SPACE = [
    ("Kernel size", "{5, 7, 9, 11, 13, 15, 21}"),
    ("Frequency band set (sets the dilation branches)", "16 definitions"),
    ("Branch combination", "concatenate or project"),
    ("Convolutional layers", "1 or 2"),
    ("Convolutional output channels", "{8, 16, 24, 32, 48}"),
    ("GRU hidden dimension", "{8, 16, 24, 32, 48, 64}"),
    ("Embedding dimension", "{8, 16, 24, 32, 48, 64}"),
    ("Convolutional stride", "2 or 4"),
    ("Normalisation layer", "none, instance, batch, or group"),
    ("Activation", "ReLU or SiLU"),
    ("CNN dropout", "0–0.20"),
    ("RNN dropout", "0–0.20"),
    ("Contrastive temperature τ", "0.5–1.2"),
    ("Gaussian age-kernel width σ", "0.3–3.5 years"),
    ("Learning rate", "5×10⁻⁴–5×10⁻³ (log scale)"),
    ("Window subsampling factor", "2 or 4"),
    ("Augmentation: noise", "0.01–0.10 of window SD"),
    ("Augmentation: amplitude scaling", "0.05–0.40"),
    ("Augmentation: time shift", "5–30 samples"),
    ("Augmentation: segment dropout", "10–80 samples"),
]

FIXED = [
    ("GRU layers", "1"),
    ("Batch size", "256"),
    ("Epochs per trial", "10"),
    ("Optimiser", "Adam, mixed precision"),
    ("Train / validation split", "75 % / 25 %, by subject"),
]


def load_fig3_rows():
    """Per-channel capacity and embedding quality, i.e. what Figure 3 plots."""
    sys.path.insert(0, HERE)
    from figures_fig3_stage2 import load as _load
    d = _load().sort_values("r", ascending=False).reset_index(drop=True)
    out = []
    for _, r in d.iterrows():
        out.append([
            r.ch,
            "/".join(str(x) for x in r.dil),
            f"{r.f_lo:.1f}\u2013{r.f_hi:.0f}",
            f"{r.rf_ms:.0f}",
            f"{int(r.conv_cap):,}",
            f"{int(r.gru_par):,}",
            f"{int(r.repr_cap):,}",
            f"{r.conv_par / r.gru_par:.3f}",
            f"{r['align']:.3f}",
            f"{r['unif']:.2f}",
            f"{r.r:.3f}",
            f"{r.slope:+.2f}",
            f"{-r.score:.3f}",
        ])
    return out


FIG3_HDR = ["Channel", "Dilations", "Passband (Hz)", "Effective RF (ms)",
            "Conv. capacity", "GRU parameters", "Repr. capacity",
            "Conv/RNN ratio", "Alignment", "Uniformity", "Age r", "Slope",
            "Information score"]


def load_rows():
    rows = []
    for ch in CHANNELS:
        ck = torch.load(os.path.join(CKPT, f"{ch}.pt"), map_location="cpu",
                        weights_only=False)
        c = ck["cfg"]
        rows.append({
            "ch": ch,
            "kernel": c["kernel_size"],
            "bands": c["band_set"],
            "dil": ", ".join(str(x) for x in c["dilation"]),
            "layers": c["arch_extra"]["conv_layers"],
            "conv": c["conv_channels"],
            "gru": c["gru_hidden"],
            "emb": c["embedding_dim"],
            "mix": c["arch_extra"]["branch_mix"],
            "norm": c["arch_extra"].get("norm", "none"),
            "stride": c["stride"],
            "act": c["activation"],
            "cdrop": c["cnn_dropout"],
            "rdrop": c["rnn_dropout"],
            "tau": c["temperature"],
            "sigma": c["sigma"],
            "lr": c["lr"],
            "wsm": c["win_stride_mult"],
            "noise": c["aug_noise"],
            "scale": c["aug_scale"],
            "shift": c["aug_shift"],
            "drop": c["aug_drop"],
            "npar": ck["n_params"],
            "epoch": ck["epoch"],
        })
    return rows


def add_table(doc, header, body, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(font)
    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


# Data partition. Window counts are what the pipeline prints at load time;
# subject counts and chance floors are recomputed from the split file and the
SPLIT = [
    # stage, partition, subjects, windows, age, chance MAE, role
    ("Encoder pre-training (public)", "Training", "539", "197,605",
     "42.9 ± 14.5", "–",
     "Contrastive objective; no age labels used as targets"),
    ("Encoder pre-training (public)", "Validation", "179", "66,523",
     "44.9 ± 14.5", "–",
     "Early stopping and learning-rate schedule"),
    ("Age model (public)", "Training", "539", "98,803", "42.9 ± 14.5",
     "12.71", "In the regression loss"),
    ("Age model (public)", "Validation", "179", "33,262", "44.9 ± 14.5",
     "12.88", "Monitoring; not in the loss"),
    ("Age model (in-house)", "Fit", "20", "3,495", "36.8 ± 11.4", "10.05",
     "In the regression loss, jointly with the public pool"),
    ("Age model (in-house)", "Validation", "20", "3,270", "37.7 ± 11.9",
     "10.40", "Epoch and configuration selection"),
    ("Age model (in-house)", "Held-out", "41", "6,217", "36.5 ± 10.7",
     "9.10", "Never in the loss; reported result"),
]

SPLIT_HDR = ["Stage", "Partition", "Subjects", "Windows", "Age (years)",
             "Chance MAE (years)", "Role"]


def main():
    rows = load_rows()
    print("building the methods tables ...")
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.7)

    # ---- Table 1: the search space ------------------------------------
    doc.add_heading("Table 1. Hyperparameter search space", level=2)
    add_table(doc, ["Parameter", "Search space"],
              [[a, b] for a, b in SPACE], widths=[3.9, 3.0], font=9)
    caption(doc, "Parameters optimised independently for each of the 14 "
                 "channels (2,314 completed trials in total, 156–184 per "
                 "channel).")
    doc.add_paragraph()
    doc.add_heading("Fixed settings", level=3)
    add_table(doc, ["Setting", "Value"], [[a, b] for a, b in FIXED],
              widths=[3.9, 3.0], font=9)

    # ---- Table 2: the data partition ----------------------------------
    doc.add_page_break()
    doc.add_heading("Table 2. Data partition", level=2)
    add_table(doc, SPLIT_HDR, [list(r) for r in SPLIT],
              widths=[1.5, 0.9, 0.7, 0.8, 0.9, 0.9, 1.7], font=8.5)
    caption(doc, "Partitions are disjoint by participant at every stage. "
                 "Chance MAE is the error of always predicting that "
                 "partition’s median age. Public windows use a 256-sample "
                 "hop; in-house windows use a 64-sample hop.")

    # ---- landscape section for the wide per-channel tables ------------
    doc.add_page_break()
    sec = doc.add_section()
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.left_margin = sec.right_margin = Inches(0.5)

    doc.add_heading("Supplementary Table 1a. Selected encoder architecture "
                    "per channel", level=2)
    add_table(doc,
              ["Channel", "Kernel", "Band set", "Dilations", "Conv layers",
               "Conv ch.", "GRU hidden", "Embedding", "Branch mix", "Norm",
               "Stride", "Parameters"],
              [[r["ch"], r["kernel"], r["bands"], r["dil"], r["layers"],
                r["conv"], r["gru"], r["emb"], r["mix"], r["norm"],
                r["stride"], f"{r['npar']:,}"] for r in rows])
    caption(doc, "Dilations are derived from the selected band set and kernel "
                 "size; each value defines one parallel convolutional branch.")

    doc.add_paragraph()
    doc.add_heading("Supplementary Table 1b. Selected training and "
                    "augmentation settings per channel", level=2)
    add_table(doc,
              ["Channel", "Activation", "CNN dropout", "RNN dropout",
               "Temperature τ", "Age kernel σ (yr)", "Learning rate",
               "Window subsampling", "Noise (× SD)", "Scaling",
               "Shift (samples)", "Dropout (samples)", "Best epoch"],
              [[r["ch"], r["act"], f"{r['cdrop']:.3f}", f"{r['rdrop']:.3f}",
                f"{r['tau']:.3f}", f"{r['sigma']:.3f}", f"{r['lr']:.2e}",
                r["wsm"], f"{r['noise']:.3f}", f"{r['scale']:.3f}",
                r["shift"], r["drop"], r["epoch"]] for r in rows])
    caption(doc, "Values are the median of each channel's top 30 % of trials "
                 "(rounded median for integers, mode for categorical "
                 "parameters). Best epoch is the early-stopping epoch of the "
                 "final training run.")

    # ---- Supplementary Table 2: what Figure 3 plots -------------------
    doc.add_paragraph()
    doc.add_heading("Supplementary Table 2. Per-channel encoder capacity and "
                    "embedding quality", level=2)
    add_table(doc, FIG3_HDR, load_fig3_rows())
    caption(doc, "Quantities plotted in Figure 3, ordered by age correlation. "
                 "Conv. capacity is kernel size \u00d7 channels; GRU parameters "
                 "is 2\u00b73\u00b7H(In+H+2); representational capacity is "
                 "projection size \u00d7 hidden size. Alignment and uniformity "
                 "are the embedding-geometry measures of Wang & Isola (2020), "
                 "with targets of 0.3 and \u22122.0 respectively. Age r is the "
                 "subject-level correlation of a ridge probe on the frozen "
                 "embedding and slope its age-bias slope. The information score "
                 "is the search objective, sign-flipped so that higher is "
                 "better.")

    # ---- Supplementary Table 3: lobe-block statistics -----------------
    bt = os.path.join(os.path.dirname(OUT), "Table_lobe_blocks.csv")
    if os.path.exists(bt):
        import csv
        with open(bt) as f:
            rr = list(csv.reader(f))
        doc.add_paragraph()
        doc.add_heading("Supplementary Table 3. Log-Euclidean block "
                        "statistics against age", level=2)
        add_table(doc, rr[0], rr[1:])
        caption(doc, "Mean off-diagonal G_log entry within and between "
                     "anatomical blocks, correlated with age across the 41 "
                     "held-out participants. ANT = AF3 AF4 F7 F3 F4 F8 FC5 "
                     "FC6; POS = P7 P8 O1 O2; TMP = T7 T8. p_perm is a "
                     "10,000-shuffle permutation test on the age labels and "
                     "p_holm its Holm correction across the six blocks. "
                     "Young / mid / old are the age-tertile means.")

    # ---- Supplementary Table 4: every PASA index tested ---------------
    pt = os.path.join(os.path.dirname(OUT),
                      "Table_PASA_ratio_scores_age_phase4_t25.csv")
    if os.path.exists(pt):
        import pandas as _pd
        pd_ = _pd.read_csv(pt)
        fam = pd_[pd_.score.str.startswith("FFT ")
                  & ~pd_.score.str.contains("brain_score")]
        sys.path.insert(0, os.path.join(HERE, "strengthen"))
        from pasa_index import holm as _h
        pd_["p_holm_bands"] = ""
        pd_.loc[fam.index, "p_holm_bands"] = [f"{v:.3f}" for v in
                                              _h(fam.p_perm.to_numpy())]
        doc.add_paragraph()
        doc.add_heading("Supplementary Table 4. Anterior/posterior indices "
                        "tested against age", level=2)
        add_table(doc,
                  ["Arm", "Index", "n", "Pearson r", "Spearman rho",
                   "p_perm", "p_holm (band family)"],
                  [[r.arm, r.score, int(r.n), f"{r.r:+.3f}",
                    f"{r.spearman:+.3f}", f"{r.p_perm:.4f}", r.p_holm_bands]
                   for r in pd_.itertuples()])
        caption(doc, "Every index computed, including those not plotted. "
                     "p_perm is a 10,000-shuffle permutation test on the age "
                     "labels; the Holm column corrects across the five "
                     "absolute-power band tests only. Indices are ratios of "
                     "anterior to posterior quantities, formed as "
                     "exp(mean log ANT \u2212 mean log POS).")

    # ---- Supplementary Table 5: per-channel correlations with age -----
    ct = os.path.join(os.path.dirname(OUT), "Table_lobe_per_channel.csv")
    if os.path.exists(ct):
        import csv as _csv
        with open(ct) as f:
            cr = list(_csv.reader(f))
        hdr = ["Channel", "Region", "G_log diag", "log-cov diag", "delta",
               "theta", "alpha", "beta", "gamma", "broadband"]
        doc.add_paragraph()
        doc.add_heading("Supplementary Table 5. Per-channel correlation with "
                        "age", level=2)
        add_table(doc, hdr[:len(cr[0])], cr[1:])
        caption(doc, "Pearson correlation of each channel\u2019s value with "
                     "age. G_log diag is the model-space diagonal over the 41 "
                     "held-out participants; all other columns are model-free "
                     "log power over all 81. Beta power declines at every "
                     "posterior channel (mean r = \u22120.237) and only "
                     "weakly at anterior channels (mean \u22120.087), so the "
                     "anterior/posterior ratio rises through posterior "
                     "decline rather than anterior increase.")

    doc.save(OUT)
    print(f"wrote {OUT}")
    print(f"  Table 1: {len(SPACE)} searched parameters, {len(FIXED)} fixed")
    print(f"  Table 2: {len(SPLIT)} partitions")
    print(f"  Supp Table 1a/1b: {len(rows)} channels")
    print(f"  Supp Table 2: {len(load_fig3_rows())} channels")


if __name__ == "__main__":
    main()
