"""Build PREPROCESSED time-domain windows for encoder training (Review-only)."""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config_review as C

FS = 256
WIN, STRIDE = 512, 256          # 2 s windows, 1 s hop (matches the sweeps' effective density)
HP, LP, NOTCH = 0.5, 45.0, 50.0
AMP_UV, GRAD_UV, EQ_MIN, MOT_Z = 100.0, 25.0, 1, 4.0
MAX_BAD_FRAC = 0.10             # drop a window if >10% of its samples are artifact
DISCARD_EDGE = 100              # rows dropped at each stimulus edge (as in 0_isolate_stimuli)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_windows_cache.pkl")

# Filtering / artifact-rejection recipes, spanning what was tried during teh
# investigation: from teh original pipeline's behaviour (none) to the full
PREPROC_RECIPES = {
    # device output as-is: no filter, no re-reference, no artifact rejection.
    # This is what the Snakemake pipeline effectively consumed.
    "raw":        dict(filter=False, hp=None, lp=None, notch=False, reref=False,
                       artifact=False, amp=None, grad=None, use_eq=False,
                       use_mot=False, eq_min=1, max_bad=1.0),
    # filtering only, no artifact rejection -> isolates the effect of filtering
    "filt_only":  dict(filter=True, hp=0.5, lp=45.0, notch=True, reref=True,
                       artifact=False, amp=None, grad=None, use_eq=False,
                       use_mot=False, eq_min=1, max_bad=1.0),
    # filtering + amplitude/gradient rejection, no device-quality channels
    "filt_amp":   dict(filter=True, hp=0.5, lp=45.0, notch=True, reref=True,
                       artifact=True, amp=100.0, grad=25.0, use_eq=False,
                       use_mot=False, eq_min=1, max_bad=0.10),
    # teh full recipe used in the re-analysis (EQ + motion gating as well)
    "filtered":   dict(filter=True, hp=0.5, lp=45.0, notch=True, reref=True,
                       artifact=True, amp=100.0, grad=25.0, use_eq=True,
                       use_mot=True, eq_min=1, max_bad=0.10),
    # aggressive: 1 Hz high-pass (kills drift harder) + stricter thresholds
    # z-score-native recipes. On per-subject z-scored data teh microvolt
    "prefiltered_uv": {"filter": False, "hp": None, "lp": None, "notch": False,
                       "reref": False, "artifact": True, "amp": 100.0, "grad": 25.0,
                       "use_eq": True, "use_mot": True, "eq_min": 1, "max_bad": 0.1},
    "prefiltered_z": {"filter": False, "hp": None, "lp": None, "notch": False,
                      "reref": False, "artifact": True, "amp": 6.0, "grad": 1.5,
                      "use_eq": True, "use_mot": True, "eq_min": 1, "max_bad": 0.1},
    "filtered_z": {"filter": True, "hp": 0.5, "lp": 45.0, "notch": True,
                   "reref": True, "artifact": True, "amp": 6.0, "grad": 1.5,
                   "use_eq": True, "use_mot": True, "eq_min": 1, "max_bad": 0.1},
    "filt_only_z": {"filter": True, "hp": 0.5, "lp": 45.0, "notch": True,
                    "reref": True, "artifact": False, "amp": None, "grad": None,
                    "use_eq": False, "use_mot": False, "eq_min": 1, "max_bad": 1.0},
    "strict":     dict(filter=True, hp=1.0, lp=40.0, notch=True, reref=True,
                       artifact=True, amp=75.0, grad=15.0, use_eq=True,
                       use_mot=True, eq_min=1, max_bad=0.05),
    # lenient: wide band, loose thresholds (keeps the most data)
    "lenient":    dict(filter=True, hp=0.1, lp=45.0, notch=True, reref=False,
                       artifact=True, amp=150.0, grad=40.0, use_eq=False,
                       use_mot=False, eq_min=0, max_bad=0.25),
}


def filters():
    return (butter(4, HP / (FS / 2), btype="highpass"),
            iirnotch(NOTCH / (FS / 2), Q=30),
            butter(4, LP / (FS / 2), btype="lowpass"))


def subject_windows(sid, channels, filt, win=None, stride=None, preproc='filtered'):
    dpath = os.path.join(ROOT, "data", sid, "data.csv")
    mpath = os.path.join(ROOT, "data", sid, "marker.csv")
    if not (os.path.exists(dpath) and os.path.exists(mpath)):
        return None
    eq_cols = [f"EQ.{c.replace('EEG.', '')}" for c in channels]
    mot_cols = ["MOT.AccX", "MOT.AccY", "MOT.AccZ"]
    keep = set(channels + eq_cols + mot_cols + ["MarkerIndex"])
    df = pd.read_csv(dpath, skiprows=1, usecols=lambda c: c in keep, low_memory=False)
    if not all(c in df.columns for c in channels):
        return None
    mk = pd.read_csv(mpath)
    try:
        mk = mk[mk[mk["type"] == "eyesopen"].index.values[0]:]
    except Exception:
        pass
    mk = mk[mk["type"] != "active_period"]

    X = df[channels].to_numpy(float)
    N = len(X)
    rec = PREPROC_RECIPES[preproc]
    if rec["filter"]:
        b_hp, a_hp = butter(4, rec["hp"] / (FS / 2), btype="highpass")
        X = filtfilt(b_hp, a_hp, X, axis=0)
        if rec["notch"]:
            b_no, a_no = iirnotch(NOTCH / (FS / 2), Q=30)
            X = filtfilt(b_no, a_no, X, axis=0)
        if rec["lp"]:
            b_lp, a_lp = butter(4, rec["lp"] / (FS / 2), btype="lowpass")
            X = filtfilt(b_lp, a_lp, X, axis=0)
    if rec["reref"]:
        X = X - X.mean(axis=1, keepdims=True)             # average reference

    bad = np.zeros_like(X, dtype=bool)
    if rec["artifact"]:
        bad |= np.abs(X) > rec["amp"]
        bad |= np.abs(np.diff(X, axis=0, prepend=X[:1])) > rec["grad"]
        if rec["use_eq"] and all(c in df.columns for c in eq_cols):
            bad |= df[eq_cols].ffill().bfill().to_numpy(float) <= rec["eq_min"]
        if rec["use_mot"] and all(c in df.columns for c in mot_cols):
            acc = df[mot_cols].ffill().bfill().to_numpy(float)
            if np.isfinite(acc).any():
                mag = np.linalg.norm(acc - np.nanmedian(acc, 0), axis=1)
                sd = np.nanstd(mag) or 1.0
                bad |= (((mag - np.nanmedian(mag)) / sd) > MOT_Z)[:, None]

    # interpolate bad samples per channel so windows stay continuous
    Xi = X.copy()
    for c in range(X.shape[1]):
        m = bad[:, c]
        if m.all():
            continue
        if m.any():
            idx = np.arange(N)
            Xi[m, c] = np.interp(idx[m], idx[~m], X[~m, c])

    # stimulus label per row from markers; FIRST marker wins (kills duplication)
    task = np.full(N, None, dtype=object)
    if "MarkerIndex" in df.columns:
        mi = df["MarkerIndex"].to_numpy()
        for mid, typ in zip(mk["marker_id"], mk["type"]):
            s = np.flatnonzero(mi == mid)
            e = np.flatnonzero(mi == -mid)
            if len(s) and len(e) and e[0] > s[0]:
                a, b = s[0] + DISCARD_EDGE, e[0] - DISCARD_EDGE
                if b > a:
                    seg = task[a:b]
                    seg[seg == None] = typ                 # noqa: E711

    win = win or WIN; stride = stride or STRIDE
    Xs, stims, bad_frac = [], [], []
    for st in range(0, N - win + 1, stride):
        sl = slice(st, st + win)
        t = task[sl]
        t0 = t[0]
        if t0 is None or not (t == t0).all():              # window must sit in one stimulus
            continue
        bf = bad[sl].mean()
        if bf > rec["max_bad"]:
            continue
        Xs.append(Xi[sl].astype(np.float32))
        stims.append(str(t0)); bad_frac.append(float(bf))
    if not Xs:
        return None
    return np.stack(Xs), np.array(stims), float(np.mean(bad_frac))


def build(channels, win=None, stride=None, cache=None, preproc='filtered'):
    win = win or WIN; stride = stride or STRIDE
    cache = cache or CACHE
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    idx = pd.read_pickle(C.MASTER_INDEX)
    ages = idx.groupby("ID")["Age"].first().to_dict()
    filt = filters()
    data = {}
    subs = sorted(ages)
    print(f"building clean windows for {len(subs)} subjects "
          f"({win}-sample / {win/FS:.1f}s, stride {stride}, {preproc})...", flush=True)
    for i, sid in enumerate(subs, 1):
        out = subject_windows(sid, channels, filt, win, stride, preproc)
        if out is None:
            print(f"  [{i}/{len(subs)}] {sid}: none", flush=True); continue
        X, stims, bf = out
        data[sid] = {"X": X, "stim": stims, "age": float(ages[sid])}
        if i % 10 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] {sid}: {len(X)} windows, "
                  f"{len(set(stims))} stimuli, mean bad {bf:.1%}", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    tot = sum(len(v["X"]) for v in data.values())
    print(f"\ncached {len(data)} subjects, {tot:,} clean windows "
          f"({tot/max(len(data),1):.0f}/subject)")
    return data


if __name__ == "__main__":
    with open(C.CONFIG_JSON) as f:
        ch = json.load(f)["EEG_columns"]
    build(ch)
