"""Unified access to all harmonised EEG datasets (Review-only)."""
import json
import os

import numpy as np
import pandas as pd

# Output/input root for the harmonised arrays. Defaults to the copy inside the
# project, so nothing that does not set the variable changes behaviour; set
HARM_DIR = os.environ.get(
    "HARMONIZED_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonized"))
DATASETS = ["ours", "ds005385", "ds003775", "ds003478", "ds004148", "ds003969"]
WIN, STRIDE, FS = 512, 256, 256


def dataset_dir(name):
    return os.path.join(HARM_DIR, name)


def load_dataset(name, mmap=True):
    """Return (X, recordings_df)."""
    d = dataset_dir(name)
    # HARMONIZED_Z=1 selects the per-subject per-channel z-scored arrays written
    # by zscore_filtered.py. Raw microvolt amplitudes are not comparable across
    if os.environ.get("HARMONIZED_F"):
        fname = "continuous_X_f.npy"       # filtered + rereferenced, NOT normalised
    elif os.environ.get("HARMONIZED_ZDR"):
        fname = "continuous_X_zdr.npy"     # filter -> per-dataset median/MAD
    elif os.environ.get("HARMONIZED_ZFR"):
        fname = "continuous_X_zfr.npy"     # filter -> per-subject median/MAD
    elif os.environ.get("HARMONIZED_ZD"):
        fname = "continuous_X_zd.npy"      # filter -> per-DATASET z-score
    elif os.environ.get("HARMONIZED_ZF"):
        fname = "continuous_X_zf.npy"      # filter -> per-SUBJECT z-score
    elif os.environ.get("HARMONIZED_Z"):
        fname = "continuous_X_z.npy"
    else:
        fname = "continuous_X.npy"
    path = os.path.join(d, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} — run zscore_filtered.py first")
    X = np.load(path, mmap_mode="r" if mmap else None)
    recs = pd.read_parquet(os.path.join(d, "recordings.parquet"))
    return X, recs


def info(name):
    with open(os.path.join(dataset_dir(name), "info.json")) as f:
        return json.load(f)


def iter_recordings(name, subjects=None, stimuli=None):
    """Yield (meta_row, [lenght, 14] array) for each recording."""
    X, recs = load_dataset(name)
    if subjects is not None:
        recs = recs[recs.subject.isin(set(subjects))]
    if stimuli is not None:
        recs = recs[recs.stimulus.isin(set(stimuli))]
    for r in recs.itertuples():
        yield r, np.asarray(X[r.start:r.start + r.length])


def build_windows(name, recipe="filtered", win=WIN, stride=STRIDE,
                  subjects=None, stimuli=None, out_path=None, verbose=True):
    """Apply a preprocessing recipe and cut fixed-length windows."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strengthen"))
    from scipy.signal import butter, filtfilt, iirnotch
    from clean_windows import PREPROC_RECIPES, NOTCH

    rec = PREPROC_RECIPES[recipe]
    Xs, subj, stim, ages = [], [], [], []
    n_rec = 0
    n_short = 0
    for meta, sig in iter_recordings(name, subjects, stimuli):
        S = sig.astype(np.float64)
        # shorter than one window yields nothing and filtfilt won't pad it,
        # so drop it before filtering instead of crashing
        if len(S) < win:
            n_short += 1
            continue
        if rec["filter"]:
            b, a = butter(4, rec["hp"] / (FS / 2), btype="highpass")
            S = filtfilt(b, a, S, axis=0)
            if rec["notch"]:
                b, a = iirnotch(NOTCH / (FS / 2), Q=30)
                S = filtfilt(b, a, S, axis=0)
            if rec["lp"]:
                b, a = butter(4, rec["lp"] / (FS / 2), btype="lowpass")
                S = filtfilt(b, a, S, axis=0)
        if rec["reref"]:
            S = S - S.mean(axis=1, keepdims=True)
        bad = np.zeros_like(S, dtype=bool)
        if rec["artifact"]:
            bad |= np.abs(S) > rec["amp"]
            bad |= np.abs(np.diff(S, axis=0, prepend=S[:1])) > rec["grad"]
            for c in range(S.shape[1]):
                m = bad[:, c]
                if m.any() and not m.all():
                    idx = np.arange(len(S))
                    S[m, c] = np.interp(idx[m], idx[~m], S[~m, c])
        for s in range(0, len(S) - win + 1, stride):
            sl = slice(s, s + win)
            if bad[sl].mean() > rec["max_bad"]:
                continue
            Xs.append(S[sl].astype(np.float32))
            subj.append(meta.subject); stim.append(meta.stimulus); ages.append(meta.age)
        n_rec += 1
    if not Xs:
        return None
    out = {"X": np.stack(Xs), "subject": np.array(subj), "stimulus": np.array(stim),
           "age": np.array(ages, dtype=float), "dataset": name, "recipe": recipe,
           "win": win, "stride": stride}
    if verbose:
        if n_short:
            print(f"{name}/{recipe}: skipped {n_short} recordings shorter "
                  f"than one {win}-sample window")
        print(f"{name}/{recipe}: {out['X'].shape[0]:,} windows from {n_rec} recordings, "
              f"{len(set(subj))} subjects")
    if out_path:
        np.save(out_path + "_X.npy", out["X"])
        pd.DataFrame({"subject": subj, "stimulus": stim, "age": ages}).to_parquet(
            out_path + "_meta.parquet")
    return out


def summary():
    rows = []
    for n in DATASETS:
        try:
            X, recs = load_dataset(n)
            rows.append({"dataset": n, "subjects": recs.subject.nunique(),
                         "recordings": len(recs),
                         "hours": recs.length.sum() / FS / 3600,
                         "age_min": recs.age.min(), "age_max": recs.age.max(),
                         "age_median": recs.age.median(),
                         "GB": X.nbytes / 1e9})
        except Exception:
            continue
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = summary()
    if len(df):
        print(df.to_string(index=False))
        print(f"\nTOTAL: {df.subjects.sum()} subjects, {df.recordings.sum()} recordings, "
              f"{df.hours.sum():.0f} h, {df.GB.sum():.1f} GB")
    else:
        print("nothing harmonised yet — run harmonize_all.py")
