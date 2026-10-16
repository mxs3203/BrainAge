"""Filter FIRST, then per-subject per-channel z-score."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "strengthen"))
from harmonized_io import DATASETS, dataset_dir, FS          # noqa: E402
from harmonized_io import HARM_DIR  # noqa: E402

# Same guard as harmonize_all.py: this writes continuous_X_f.npy next to the base
# arrays, which is the file every downstream stage actually reads, so an accidental
_PROJECT_HARM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonized")
if (os.path.abspath(HARM_DIR) == os.path.abspath(_PROJECT_HARM)
        and os.environ.get("HARMONIZE_ALLOW_PROJECT") != "1"):
    raise SystemExit(
        "REFUSING to write into the project copy:\n  " + _PROJECT_HARM +
        "\nSet HARMONIZED_DIR=<other path>, or HARMONIZE_ALLOW_PROJECT=1 to override.")
from clean_windows import NOTCH                               # noqa: E402

EPS = 1e-8
HP, LP = 0.5, 45.0
# ZSCORE_LEVEL=subject  one mean/std per participant. Harmonises hardest but
#                       removes BETWEEN-SUBJECT amplitude differences -- wich is
LEVEL = os.environ.get("ZSCORE_LEVEL", "subject")
# ZSCORE_ROBUST=1 uses median / MAD instead of mean / std. Measured reason: with
# mean/std, a subject's scale is set by their WORST moments -- artifacts, movement,
ROBUST = bool(os.environ.get("ZSCORE_ROBUST"))
# level "none" writes the filtered, re-referenced signal with NO normalisation.
# The large search needs this: it lets window size and normalisation vary as
OUT_NAME = {("none", False): "continuous_X_f.npy",
            ("none", True):  "continuous_X_f.npy",
            ("subject", False): "continuous_X_zf.npy",
            ("dataset", False): "continuous_X_zd.npy",
            ("subject", True):  "continuous_X_zfr.npy",
            ("dataset", True):  "continuous_X_zdr.npy"}[(LEVEL, ROBUST)]
MAD_K = 1.4826          # MAD -> sigma for a normal distribution
SUBSAMPLE = 400_000     # values per channel for the median estimate


def centre_scale(vals):
    """vals [n, channels] -> (centre, scale) per channel."""
    if ROBUST:
        med = np.median(vals, axis=0)
        mad = np.median(np.abs(vals - med), axis=0) * MAD_K
        return med, np.where(mad < EPS, 1.0, mad)
    mu = vals.mean(0)
    sd = vals.std(0)
    return mu, np.where(sd < EPS, 1.0, sd)
B_HP = butter(4, HP / (FS / 2), btype="highpass")
B_NO = iirnotch(NOTCH / (FS / 2), Q=30)
B_LP = butter(4, LP / (FS / 2), btype="lowpass")


def filt(S):
    S = filtfilt(*B_HP, S, axis=0)
    S = filtfilt(*B_NO, S, axis=0)
    S = filtfilt(*B_LP, S, axis=0)
    return S - S.mean(axis=1, keepdims=True)      # common average reference


def run(name):
    d = dataset_dir(name)
    X = np.load(os.path.join(d, "continuous_X.npy"), mmap_mode="r")   # always RAW
    recs = pd.read_parquet(os.path.join(d, "recordings.parquet"))
    out = np.lib.format.open_memmap(os.path.join(d, OUT_NAME),
                                    mode="w+", dtype=np.float32, shape=X.shape)
    stats, covered = {}, 0
    if LEVEL == "none":
        for r in recs.itertuples():
            a, b = int(r.start), int(r.start + r.length)
            if b - a < 64:
                out[a:b] = 0.0; continue
            out[a:b] = filt(np.asarray(X[a:b], dtype=np.float64)).astype(np.float32)
            covered += b - a
        out.flush()
        print(f"  {name:10s} filtered only | {covered:,}/{X.shape[0]:,} samples", flush=True)
        return
    if LEVEL == "dataset":
        # one pass over every recording to get cohort-level statistics, then a
        # second pass applying the SAME constants to every subject, so relative
        chunks, cnt = [], 0
        nrec = max(1, len(recs))
        for r in recs.itertuples():
            a, b = int(r.start), int(r.start + r.length)
            if b - a < 64:
                continue
            F = filt(np.asarray(X[a:b], dtype=np.float64))
            cnt += len(F)
            step = max(1, len(F) // max(1, SUBSAMPLE // nrec))
            chunks.append(F[::step])
        V = np.concatenate(chunks)
        if len(V) > SUBSAMPLE:
            V = V[np.linspace(0, len(V) - 1, SUBSAMPLE).astype(int)]
        g_mu, g_sd = centre_scale(V)
        for r in recs.itertuples():
            a, b = int(r.start), int(r.start + r.length)
            if b - a < 64:
                out[a:b] = 0.0; continue
            F = filt(np.asarray(X[a:b], dtype=np.float64))
            out[a:b] = ((F - g_mu) / g_sd).astype(np.float32)
            covered += b - a
        out.flush()
        json.dump({"level": "per-dataset per-channel, AFTER filtering",
                   "statistic": "median/MAD" if ROBUST else "mean/std",
                   "band": [HP, LP], "notch": NOTCH, "reref": "common average",
                   "mean": g_mu.round(5).tolist(), "std": g_sd.round(5).tolist(),
                   "n_samples": int(cnt)},
                  open(os.path.join(d, ("zscore_dataset_robust_stats.json" if ROBUST else "zscore_dataset_stats.json")), "w"))
        print(f"  {name:10s} dataset-level | {covered:,}/{X.shape[0]:,} samples",
              flush=True)
        return
    for sid, g in recs.groupby("subject", sort=True):
        spans = [(int(r.start), int(r.start + r.length)) for r in g.itertuples()]
        # pass 1 - filter each recording, collect a subsample for the statistics.
        # A subsample (not streaming moments) because teh median needs the values,
        chunks, cnt = [], 0
        for a, b in spans:
            if b - a < 64:            # filtfilt needs more samples than the order
                continue
            F = filt(np.asarray(X[a:b], dtype=np.float64))
            cnt += len(F)
            step = max(1, len(F) // max(1, SUBSAMPLE // max(1, len(spans))))
            chunks.append(F[::step])
        if cnt == 0:
            continue
        V = np.concatenate(chunks)
        if len(V) > SUBSAMPLE:
            V = V[np.linspace(0, len(V) - 1, SUBSAMPLE).astype(int)]
        mu, sd_safe = centre_scale(V)
        sd = sd_safe
        # pass 2 - refilter and write normalised
        for a, b in spans:
            if b - a < 64:
                out[a:b] = 0.0
                continue
            F = filt(np.asarray(X[a:b], dtype=np.float64))
            out[a:b] = ((F - mu) / sd_safe).astype(np.float32)
            covered += b - a
        stats[str(sid)] = {"mean": mu.round(5).tolist(), "std": sd.round(5).tolist(),
                           "n_samples": int(cnt),
                           "flat_channels": int((sd < EPS).sum())}
    out.flush()
    json.dump({"level": "per-subject per-channel, AFTER filtering",
               "statistic": "median/MAD" if ROBUST else "mean/std",
               "band": [HP, LP], "notch": NOTCH, "reref": "common average",
               "n_subjects": len(stats), "subjects": stats},
              open(os.path.join(d, ("zscore_filtered_robust_stats.json" if ROBUST else "zscore_filtered_stats.json")), "w"))
    print(f"  {name:10s} {len(stats):4d} subjects | {covered:,}/{X.shape[0]:,} samples",
          flush=True)


def verify():
    # Report ROBUST spread statistics. std is outlier-dominated: the corpus
    # contains artifacts up to 63,000x teh signal scale, so std says 154x spread
    print("\nafter filter-then-normalise (median|x|~0.67, IQR~1.35 for unit scale):")
    rng = np.random.default_rng(0)
    med = {}
    for name in DATASETS:
        Z = np.load(os.path.join(dataset_dir(name), OUT_NAME), mmap_mode="r")
        idx = rng.choice(len(Z) - 512, 3000, replace=False)
        S = np.stack([np.asarray(Z[i:i + 512]) for i in np.sort(idx)])
        a = np.abs(S)
        med[name] = float(np.median(a))
        print(f"  {name:10s} median|x| {np.median(a):6.3f}  IQR "
              f"{np.subtract(*np.percentile(S,[75,25])):6.3f}  "
              f"std {S.std():9.2f}  max|x| {a.max():9.1f}")
    v = np.array(list(med.values()))
    print(f"\nspread across datasets (median|x|): {v.min():.3f} - {v.max():.3f} "
          f"= {v.max()/v.min():.2f}x")
    pub = [x for k, x in med.items() if k != "ours"]
    print(f"ours vs public: {med['ours']/np.median(pub):.2f}x")


if __name__ == "__main__":
    print(f"filter -> per-{LEVEL} "
          f"{'median/MAD (robust)' if ROBUST else 'mean/std'} -> {OUT_NAME}\n")
    for n in DATASETS:
        run(n)
    verify()
