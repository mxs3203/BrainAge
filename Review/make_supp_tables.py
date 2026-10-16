"""Supplementary tables as a Word file, numbered to match the manuscript.

USAGE: python make_supp_tables.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
import kounios_svr_baseline as K


def add_table(doc, header, rows, widths=None, font=8.0):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = str(h)
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True; r.font.size = Pt(font)
    for row in rows:
        cs = t.add_row().cells
        for i, v in enumerate(row):
            cs[i].text = "" if v is None else str(v)
            for p in cs[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.italic = True; r.font.size = Pt(8.5)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph()


def main():
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.6)

    # ---- S1  stimulus protocol -------------------------------------
    doc.add_heading("Supplementary Table 1. Stimulus protocol", level=2)
    d = pd.read_csv(os.path.join(ROOT, "stimuli_descriptions_and_expectations.csv"))
    rows = [[r.original_stimuli_name, r.primary_lobe_targeted,
             r.secondary_lobe_targeted, r.cognitive_domain,
             str(r.expected_eeg_signature)[:88]] for r in d.itertuples()]
    add_table(doc, ["Stimulus", "Primary lobe", "Secondary lobe",
                    "Cognitive domain", "Expected EEG signature"],
              rows, widths=[1.5, 0.9, 0.9, 1.5, 2.6], font=7.0)
    caption(doc, f"The {len(d)} stimulus conditions of the in-house protocol, with the "
                 "lobe each was designed to engage. Analyses in this paper use the "
                 "eyes-closed conditions only.")

    # ---- S2  per-channel frozen hyperparameters --------------------
    doc.add_heading("Supplementary Table 2. Per-channel encoder hyperparameters", level=2)
    h = pd.read_csv(os.path.join(RES, "SuppT1_frozen_hparams.csv"))
    a = pd.read_csv(os.path.join(RES, "Table_encoders_stage2.csv"))
    m = h.merge(a[["ch", "n_trials", "k", "dilations", "passband_Hz", "conv_ch",
                   "conv_layers", "gru", "emb", "params"]],
                left_on="channel", right_on="ch", how="left")
    rows = []
    for r in m.itertuples():
        rows.append([r.channel, int(r.n_trials_stage1 + (r.n_trials or 0)),
                     r.k, r.dilations, r.passband_Hz, r.conv_ch, r.conv_layers,
                     r.gru, r.emb, r.stride, r.activation,
                     f"{r.temperature:.2f}", f"{r.sigma:.2f}",
                     f"{r.lr:.1e}", f"{r.cnn_dropout:.3f}", f"{r.rnn_dropout:.3f}",
                     f"{r.aug_noise:.3f}", f"{r.aug_scale:.3f}", int(r.aug_shift),
                     int(r.aug_drop), int(r.params)])
    add_table(doc, ["Ch", "Trials", "Kernel", "Dilations", "Passband (Hz)", "Conv ch",
                    "Conv layers", "GRU", "Emb", "Stride", "Act", "tau", "sigma", "LR",
                    "CNN drop", "RNN drop", "Noise", "Scale", "Shift", "Drop", "Params"],
              rows, font=6.0)
    caption(doc, "Values frozen after the per-channel search and used for all subsequent "
                 "training. Trials is the total number of completed trials for that "
                 "channel across both search passes. Passband is the frequency range "
                 "spanned by the branch dilations; Supplementary Figure 10 shows what the "
                 "trained filters are actually selective for.")

    # ---- S3  block-level correlations ------------------------------
    doc.add_heading("Supplementary Table 3. Block-level correlations with age", level=2)
    b = pd.read_csv(os.path.join(RES, "Table_lobe_blocks.csv"))
    rows = [[r.block, f"{r.r:+.3f}", f"{r.rho:+.3f}", f"{r.p_perm:.4f}",
             f"{r.p_holm:.3f}", f"{r.young:+.4f}", f"{r.mid:+.4f}", f"{r.old:+.4f}"]
            for r in b.itertuples()]
    add_table(doc, ["Block", "Pearson r", "Spearman rho", "p (perm)", "p (Holm)",
                    "Young", "Mid", "Old"], rows, font=8.0)
    caption(doc, "Mean log-Euclidean channel-interaction strength within and between "
                 "anatomical blocks, correlated with age across the 41 held-out "
                 "participants. p (perm) is from 10,000 permutations of the age labels; "
                 "p (Holm) corrects across the six blocks. No block survives correction. "
                 "Young/Mid/Old are the tertile means.")

    # ---- S4  frontal/posterior indices -----------------------------
    doc.add_heading("Supplementary Table 4. Anterior/posterior indices and age", level=2)
    p = pd.read_csv(os.path.join(RES, "SuppT4_pasa_indices.csv"))
    rows = [[r.arm, r.index, int(r.n), f"{r.r:+.3f}",
             f"{r.spearman:+.3f}", f"{r.p_perm:.4f}"] for r in p.itertuples()]
    add_table(doc, ["Arm", "Index", "n", "Pearson r", "Spearman rho", "p (perm)"],
              rows, widths=[0.8, 3.0, 0.5, 0.9, 1.0, 0.8], font=7.5)
    caption(doc, "Every anterior/posterior index reported against continuous age, across "
                 "the 81 in-house participants. The model arm is the anterior-minus-"
                 "posterior node strength of the learned graph; the spectral arms are "
                 "computed from the recordings without the model. Each index is a "
                 "difference of log-scale quantities, that is, the logarithm of an "
                 "anterior-to-posterior ratio. p is from 10,000 permutations; Holm "
                 "correction across the five absolute-power bands gives p = 0.021 for "
                 "beta.")

    # ---- S5  per-channel correlations ------------------------------
    doc.add_heading("Supplementary Table 5. Per-channel correlations with age", level=2)
    c = pd.read_csv(os.path.join(RES, "SuppT5_per_channel.csv"))
    cols = [x for x in c.columns if x not in ("channel", "region")]
    rows = [[r[1]["channel"], r[1]["region"]] +
            [f"{r[1][x]:+.3f}" for x in cols] for r in c.iterrows()]
    add_table(doc, ["Channel", "Region"] + [x.replace("r_", "") for x in cols],
              rows, font=7.0)
    caption(doc, "Correlation with age of each channel's band power, across the 81 "
                 "analysed participants. Beta falls at every posterior electrode. The "
                 "per-electrode values for the learned graph are given in Supplementary "
                 "Figure 12C and are computed on all 799 participants, so they are not "
                 "listed alongside these.")

    # ---- S6  the classical feature inventory -----------------------
    doc.add_heading("Supplementary Table 6. Classical feature set", level=2)
    z = np.load(os.path.join(RES, "kounios_features.npz"), allow_pickle=True)
    nm = [str(x) for x in z["names"]]
    cnt = Counter(x.split("_")[0] for x in nm)
    FAM = [
        ("pow", "Absolute band power",
         "log10 of the mean periodogram over the band's frequency bins",
         "14 electrodes", "numpy.fft.rfft, numpy.hanning"),
        ("relpow", "Relative band power",
         "band power divided by the sum over all nine bands",
         "14 electrodes", "numpy"),
        ("coh", "Magnitude-squared coherence",
         "|Sxy|^2 / (Sxx Syy), averaged over the band's bins",
         "91 electrode pairs", "numpy.fft.rfft (cross-spectra)"),
        ("cv", "Band-power variability",
         "coefficient of variation (SD / mean) of band power across epochs",
         "14 electrodes", "numpy"),
        ("noise", "Spectral noise residual",
         "mean residual of the log-log PSD from its 1/f least-squares line "
         "fitted over 1-50 Hz", "14 electrodes", "numpy.polyfit"),
    ]
    rows = [[d, f, sites, "9", str(cnt[k]), lib] for k, d, f, sites, lib in FAM]
    rows.append(["Total", "", "", "", str(len(nm)), ""])
    add_table(doc, ["Feature family", "Computed as", "Sites", "Bands",
                    "Features", "Computed with"],
              rows, widths=[1.3, 2.5, 1.1, 0.5, 0.7, 1.5], font=7.0)
    caption(doc, "The 1,323 features of the classical pipeline, reconstructed from the "
                 "four families described by Kounios et al. (DOI: 10.3389/fnrgo.2024.1340732). "
                 "Spectra were estimated from 4.0-second epochs (1,024 samples at 256 Hz) "
                 "with 50% overlap and a Hann window, averaged over up to 400 epochs per "
                 "participant and never crossing a recording boundary; participants with "
                 "fewer than four epochs were dropped. Every feature is computed for each "
                 "of the nine bands in the next table. The complete list of 1,323 feature "
                 "names is given in Supplementary Data 1. Software: Python 3.12.3, "
                 "numpy 2.5.3 (spectral estimation and all feature computation), "
                 "mne 1.13.2 (channel selection and resampling to 256 Hz during "
                 "harmonisation), scipy 1.16.2 (filtering), scikit-learn 1.7.2 "
                 "(standardisation, univariate feature selection and the three "
                 "regression models), pandas 2.3.3. "
                 "No dedicated feature-extraction toolbox was used; every feature is "
                 "computed from the Hann-windowed FFT described above.")

    # ---- the nine bands -------------------------------------------
    doc.add_heading("Supplementary Table 7. Frequency bands of the classical "
                    "feature set", level=2)
    add_table(doc, ["Band", "Range (Hz)", "Features per band"],
              [[b, f"{lo}-{hi}", "147"] for b, lo, hi in K.BANDS],
              widths=[1.1, 1.1, 1.3], font=7.5)
    caption(doc, "Nine bands spanning 1-50 Hz, the range the recordings were filtered to. "
                 "Each band contributes 14 + 14 + 91 + 14 + 14 = 147 features, and "
                 "9 x 147 = 1,323.")

    pd.DataFrame({"feature": nm}).to_csv(
        os.path.join(RES, "Supplementary_Data_1_feature_names.csv"), index=False)

    out = os.path.join(RES, "Supplementary_tables.docx")
    doc.save(out)
    print("wrote", out)
    print("  S1 stimulus protocol      :", len(d), "rows")
    print("  S2 per-channel hparams    :", len(m), "rows")
    print("  S3 block correlations     :", len(b), "rows")
    print("  S4 anterior/posterior idx :", len(p), "rows")
    print("  S5 per-channel age corr   :", len(c), "rows")
    print("  S6 feature families       :", len(FAM), "rows,", len(nm), "features")
    print("  S7 frequency bands        :", len(K.BANDS), "rows")


if __name__ == "__main__":
    main()
