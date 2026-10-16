"""Harmonise the in-house EPOC data and all 5 public datasets into one format."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strengthen"))
sys.path.insert(0, os.path.dirname(HERE))
import config_review as C
from harmonized_io import HARM_DIR, FS

# Refuse to write into the project's harmonised copy unless explicitly allowed.
# Rebuilding takes hours, and every downstream artefact -- the 14 encoders, both
_PROJECT_HARM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonized")
if (os.path.abspath(os.environ.get("HARMONIZED_DIR", _PROJECT_HARM))
        == os.path.abspath(_PROJECT_HARM)
        and os.environ.get("HARMONIZE_ALLOW_PROJECT") != "1"):
    raise SystemExit(
        "REFUSING to write into the project copy:\n  " + _PROJECT_HARM +
        "\nSet HARMONIZED_DIR=<other path>, or HARMONIZE_ALLOW_PROJECT=1 to override.")

PUBLIC_ROOT = "/home/mateo/PublicEEG"
EPOC = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
        "FC6", "F4", "F8", "AF4"]
MIN_SEC = 4.0          # skip recordings shorter than this

# Only stimuli documented in stimuli_descriptions_and_expectations.csv are real
# experimental conditions. marker.csv also contains:
STIM_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "stimuli_descriptions_and_expectations.csv")
# the CSV writes this one in the singular; the marker file uses the plural
STIM_ALIAS = {"instructions_listening": "instruction_listening"}

# Between-stimulus periods (participants talking or resting) are usable data, but
# only after teh headset was set up: the first minutes are calibration noise.
INSTRUCTION_PAT = ("instruction", "insturction", "intsruction")


# Markers whose NAME contains "instructions" but which are real experimental
# conditions, not instruction periods (confirmed by the experimenter).
NOT_INSTRUCTION = {"paretial_instructions"}


def is_instruction(t):
    tl = t.lower()
    if tl in NOT_INSTRUCTION:
        return False
    return any(p in tl for p in INSTRUCTION_PAT)


SETUP_SKIP_SEC = 240.0     # calibration ends ~236 s in; skip everything before
GAP_MIN_SEC = 10.0         # a gap must be long enough to yield useful windows


def valid_stimuli():
    names = set(pd.read_csv(STIM_CSV)["original_stimuli_name"].dropna().str.strip())
    return names


def _norm(ch):
    return (ch.upper().replace("EEG ", "").replace("-REF", "")
            .replace("-LE", "").replace("-A1", "").strip())


TARGET_UV = 15.0        # physiological scalp-EEG robust std after 0.5-45 Hz band-pass


def robust_uv_std(X, fs=FS):
    """Median across channels of a robust (MAD-based) std after band-passing.
    DC and drift are removed first so electrode offset does not dominate."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [0.5 / (fs / 2), 45.0 / (fs / 2)], btype="bandpass")
    Y = filtfilt(b, a, np.asarray(X, float), axis=0)
    mad = np.median(np.abs(Y - np.median(Y, axis=0)), axis=0) * 1.4826
    return float(np.median(mad))


def calibrate_scale(sample_arrays):
    """Power-of-ten factor bringing a dataset into physiological range."""
    vals = [robust_uv_std(X) for X in sample_arrays if X is not None and len(X) > FS]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if not vals:
        return 1.0, None
    obs = float(np.median(vals))
    factor = 10.0 ** round(np.log10(TARGET_UV / obs))
    return factor, obs


def read_public_recording(path):
    """EDF/BDF/SET -> [T,14] float32 @256 Hz in microvolts, EPOC channel order."""
    import mne
    mne.set_log_level("ERROR")
    ext = os.path.splitext(path)[1].lower()
    reader = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
              ".set": mne.io.read_raw_eeglab, ".vhdr": mne.io.read_raw_brainvision}.get(ext)
    if reader is None:
        return None
    raw = reader(path, preload=True, verbose=False)
    lut = {}
    for c in raw.ch_names:
        lut.setdefault(_norm(c), c)
    if not all(e in lut for e in EPOC):
        return None
    raw.pick([lut[e] for e in EPOC])
    raw.reorder_channels([lut[e] for e in EPOC])
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raw.resample(FS, npad="auto")
    return np.ascontiguousarray(raw.get_data().T, dtype=np.float32)


def public_label(path):
    b = os.path.basename(path)
    parts = [p for p in b.split("_") if p.startswith(("task-", "acq-", "ses-", "run-"))]
    return "_".join(p.split("-", 1)[1] for p in parts) or "rest"


def harmonize_public(ds):
    root = os.path.join(PUBLIC_ROOT, ds)
    if not os.path.isdir(root):
        return None
    ages = {}
    ptsv = os.path.join(root, "participants.tsv")
    if os.path.exists(ptsv):
        df = pd.read_csv(ptsv, sep="\t")
        acol = next((c for c in df.columns if c.strip().lower() in
                     ("age", "age_years", "participant_age")), None)
        if acol:
            for _, r in df.iterrows():
                try:
                    a = float(r[acol])
                    if np.isfinite(a) and 5 < a < 110:
                        ages[str(r["participant_id"])] = a
                except (ValueError, TypeError, KeyError):
                    pass
    files = []
    for ext in ("edf", "bdf", "set", "vhdr"):
        files += glob.glob(os.path.join(root, "sub-*", "**", f"*_eeg.{ext}"), recursive=True)
    files = sorted(files)
    # --- calibrate units on a sample of recordings ---
    step = max(1, len(files) // 25)
    sample = []
    for p in files[::step][:25]:
        try:
            Xs = read_public_recording(p)
            if Xs is not None:
                sample.append(Xs[:FS * 60])
        except Exception:
            pass
    scale, obs = calibrate_scale(sample)
    print(f"    unit calibration: observed {obs:.3g} -> scale x{scale:g} "
          f"(target ~{TARGET_UV} uV)", flush=True)
    del sample

    sigs, rows, off = [], [], 0
    for i, p in enumerate(files, 1):
        sid = os.path.basename(p).split("_")[0]
        if sid not in ages:
            continue
        try:
            X = read_public_recording(p)
        except Exception:
            continue
        if X is None or len(X) < MIN_SEC * FS:
            continue
        X = (X * scale).astype(np.float32)
        sigs.append(X)
        rows.append({"dataset": ds, "subject": sid, "stimulus": public_label(p),
                     "age": ages[sid], "start": off, "length": len(X),
                     "source_file": os.path.relpath(p, root)})
        off += len(X)
        if i % 250 == 0:
            print(f"    [{i}/{len(files)}] {sid} ({off/FS/3600:.1f} h)", flush=True)
    return sigs, rows, {"source": f"OpenNeuro {ds} (CC0)", "n_files": len(files),
                        "unit_scale_applied": scale,
                        "observed_uv_std_before_scale": obs}


def harmonize_ours():
    """In-house EPOC data: one recording per (subject, stimulus) contiguous segment."""
    from clean_windows import ROOT as BA_ROOT, DISCARD_EDGE
    with open(C.CONFIG_JSON) as f:
        channels = json.load(f)["EEG_columns"]
    idx = pd.read_pickle(C.MASTER_INDEX)
    ages = idx.groupby("ID")["Age"].first().to_dict()
    VALID = valid_stimuli()
    print(f"    keeping only the {len(VALID)} documented stimuli "
          f"(+alias {list(STIM_ALIAS)})", flush=True)
    sigs, rows, off = [], [], 0
    subs = sorted(ages)
    for i, sid in enumerate(subs, 1):
        dp = os.path.join(BA_ROOT, "data", sid, "data.csv")
        mp = os.path.join(BA_ROOT, "data", sid, "marker.csv")
        if not (os.path.exists(dp) and os.path.exists(mp)):
            continue
        df = pd.read_csv(dp, skiprows=1,
                         usecols=lambda c: c in set(channels + ["MarkerIndex"]),
                         low_memory=False)
        if not all(c in df.columns for c in channels):
            continue
        mk = pd.read_csv(mp)
        try:
            mk = mk[mk[mk["type"] == "eyesopen"].index.values[0]:]
        except Exception:
            pass
        mk = mk[mk["type"] != "active_period"]
        X = df[channels].to_numpy(np.float32)      # allready 256 Hz, microvolts
        mi = df["MarkerIndex"].to_numpy() if "MarkerIndex" in df.columns else None
        if mi is None:
            continue
        used = np.zeros(len(X), dtype=bool)
        # two passes: documented stimuli first (they own their time ranges),
        # then the leftover phase_* gaps
        pass1 = [(m, STIM_ALIAS.get(str(t).strip(), str(t).strip()))
                 for m, t in zip(mk["marker_id"], mk["type"])
                 if STIM_ALIAS.get(str(t).strip(), str(t).strip()) in VALID
                 or str(t).strip().lower() in NOT_INSTRUCTION]
        claimed = {t for _, t in pass1}
        pass_instr = [(m, str(t).strip()) for m, t in zip(mk["marker_id"], mk["type"])
                      if is_instruction(str(t).strip())
                      and STIM_ALIAS.get(str(t).strip(), str(t).strip()) not in VALID]
        pass2 = []
        for m, t, lat, dur in zip(mk["marker_id"], mk["type"],
                                  mk.get("latency", pd.Series([np.nan] * len(mk))),
                                  mk.get("duration", pd.Series([np.nan] * len(mk)))):
            t = str(t).strip()
            if not t.startswith("phase_") or "calibration" in t.lower():
                continue
            if not (np.isfinite(lat) and np.isfinite(dur)):
                continue
            if lat < SETUP_SKIP_SEC or dur < GAP_MIN_SEC:
                continue                               # setup noise / too short
            pass2.append((m, t))
        for mid, typ in pass1 + pass_instr + pass2:
            s_ = np.flatnonzero(mi == mid); e_ = np.flatnonzero(mi == -mid)
            if not (len(s_) and len(e_) and e_[0] > s_[0]):
                continue
            a, b = s_[0] + DISCARD_EDGE, e_[0] - DISCARD_EDGE
            if b - a < MIN_SEC * FS or used[a:b].any():
                continue                            # first marker wins: no duplication
            used[a:b] = True
            seg = X[a:b]
            sigs.append(seg)
            rows.append({"dataset": "ours", "subject": sid, "stimulus": str(typ),
                         "age": float(ages[sid]), "start": off, "length": len(seg),
                         "source_file": f"data/{sid}/data.csv"})
            off += len(seg)
        if i % 20 == 0 or i == len(subs):
            print(f"    [{i}/{len(subs)}] {sid} ({off/FS/3600:.1f} h)", flush=True)
    return sigs, rows, {"source": "in-house EMOTIV EPOC X, 256 Hz native",
                        "n_files": len(subs), "unit_scale_applied": 1.0,
                        "observed_uv_std_before_scale": None,
                        "stimulus_filter": "stimuli_descriptions_and_expectations.csv",
                        "n_valid_stimuli": len(VALID),
                        "instructions_included": "named instruction markers "
                                                "(all spelling variants)",
                        "gaps_included": f"unclaimed phase_* >={GAP_MIN_SEC}s starting "
                                         f"after {SETUP_SKIP_SEC}s",
                        "excluded": "calibration/setup, phase_* overlapping a real "
                                    "stimulus, instructions_*, meta markers"}


def write(name, sigs, rows, extra):
    d = os.path.join(HARM_DIR, name)
    os.makedirs(d, exist_ok=True)
    X = np.concatenate(sigs) if sigs else np.zeros((0, 14), np.float32)
    np.save(os.path.join(d, "continuous_X.npy"), X)
    recs = pd.DataFrame(rows)
    recs.to_parquet(os.path.join(d, "recordings.parquet"))
    info = {"dataset": name, "fs": FS, "channels": EPOC, "units": "microvolts",
            "n_subjects": int(recs.subject.nunique()) if len(recs) else 0,
            "n_recordings": len(recs), "hours": float(X.shape[0] / FS / 3600),
            "age_min": float(recs.age.min()) if len(recs) else None,
            "age_max": float(recs.age.max()) if len(recs) else None,
            "preprocessing": "channel-pick + resample to 256 Hz only; "
                             "filtering/re-referencing applied later per recipe",
            **extra}
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  -> {name}: {info['n_subjects']} subjects, {info['n_recordings']} recordings, "
          f"{info['hours']:.1f} h, {X.nbytes/1e9:.2f} GB", flush=True)
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(HARM_DIR, exist_ok=True)
    targets = a.only or ["ours", "ds005385", "ds003775", "ds003478", "ds004148", "ds003969"]
    index = {}
    for name in targets:
        out = os.path.join(HARM_DIR, name, "continuous_X.npy")
        if os.path.exists(out) and not a.force:
            print(f"{name}: already harmonised — skip"); continue
        print(f"\n=== {name} ===", flush=True)
        res = harmonize_ours() if name == "ours" else harmonize_public(name)
        if not res:
            print(f"  {name}: not available — skipped"); continue
        sigs, rows, extra = res
        if not rows:
            print(f"  {name}: no usable recordings"); continue
        index[name] = write(name, sigs, rows, extra)
        del sigs
    ip = os.path.join(HARM_DIR, "index.json")
    old = {}
    if os.path.exists(ip):
        with open(ip) as f:
            old = json.load(f)
    old.update(index)
    with open(ip, "w") as f:
        json.dump(old, f, indent=2)
    print(f"\nwrote {ip}")


if __name__ == "__main__":
    main()
