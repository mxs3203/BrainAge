"""Does resampling to 256 Hz destroy anything in the band we analyse?"""
import glob
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EPOC = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
        "FC6", "F4", "F8", "AF4"]
BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 12), ("betaL", 12, 20),
         ("betaH", 20, 30), ("gamma", 30, 45)]
TARGET = 256.0
N_REC = 4


def _norm(c):
    return c.upper().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def band_power(x, fs):
    """x [T, C] -> dict band -> mean power across channels, uV^2/Hz."""
    n = int(fs * 4)                       # 4-s windows, same for every rate
    nseg = len(x) // n
    if nseg < 2:
        return None
    seg = x[:nseg * n].reshape(nseg, n, x.shape[1])
    w = np.hanning(n)[None, :, None]
    F = np.fft.rfft(seg * w, axis=1)
    P = (F.real ** 2 + F.imag ** 2) / (fs * (w ** 2).sum())
    f = np.fft.rfftfreq(n, 1 / fs)
    out = {}
    for name, lo, hi in BANDS:
        s = (f >= lo) & (f < hi)
        out[name] = float(P[:, s].mean())
    out["_sd"] = float(x.std())
    return out


def load_native(path):
    import mne
    mne.set_log_level("ERROR")
    rd = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
          ".set": mne.io.read_raw_eeglab, ".vhdr": mne.io.read_raw_brainvision}
    raw = rd[os.path.splitext(path)[1].lower()](path, preload=True, verbose=False)
    lut = {}
    for c in raw.ch_names:
        lut.setdefault(_norm(c), c)
    if not all(e in lut for e in EPOC):
        return None
    raw.pick([lut[e] for e in EPOC])
    raw.reorder_channels([lut[e] for e in EPOC])
    return raw


def methods(raw):
    import mne
    from scipy.signal import resample_poly, decimate
    fs = raw.info["sfreq"]
    out = {}

    r = raw.copy(); r.resample(TARGET, npad="auto")
    out["fft"] = r.get_data().T * 1e6

    try:
        r = raw.copy(); r.resample(TARGET, method="polyphase")
        out["polyphase"] = r.get_data().T * 1e6
    except (TypeError, ValueError) as e:
        out["polyphase"] = None

    fr = Fraction(TARGET / fs).limit_denominator(1000)
    x = raw.get_data().T * 1e6
    out["resample_poly"] = resample_poly(x, fr.numerator, fr.denominator, axis=0)

    r = raw.copy(); r.filter(None, 45.0, fir_design="firwin", verbose=False)
    r.resample(TARGET, npad="auto")
    out["lp45_then_fft"] = r.get_data().T * 1e6

    if abs(fs / TARGET - round(fs / TARGET)) < 1e-9:
        out["decimate_cheby"] = decimate(x, int(round(fs / TARGET)), ftype="iir",
                                         axis=0, zero_phase=True)
    else:
        out["decimate_cheby"] = None
    return out


def main():
    rows = []
    for ds in ("ds005385", "ds003775"):
        fs_files = sorted(glob.glob(f"/home/mateo/PublicEEG/{ds}/sub-*/**/*_eeg.*",
                                    recursive=True))
        fs_files = [f for f in fs_files
                    if os.path.splitext(f)[1].lower() in (".edf", ".bdf", ".set", ".vhdr")]
        done = 0
        for path in fs_files:
            if done >= N_REC:
                break
            raw = load_native(path)
            if raw is None:
                continue
            native_fs = raw.info["sfreq"]
            ref = band_power(raw.get_data().T * 1e6, native_fs)
            if ref is None:
                continue
            for name, x in methods(raw).items():
                if x is None:
                    continue
                bp = band_power(np.ascontiguousarray(x), TARGET)
                if bp is None:
                    continue
                for b, _, _ in BANDS:
                    rows.append({"dataset": ds, "native_fs": native_fs,
                                 "recording": os.path.basename(path),
                                 "method": name, "band": b,
                                 "ratio_vs_native": bp[b] / max(ref[b], 1e-20)})
                rows.append({"dataset": ds, "native_fs": native_fs,
                             "recording": os.path.basename(path),
                             "method": name, "band": "broadband_SD",
                             "ratio_vs_native": bp["_sd"] / max(ref["_sd"], 1e-20)})
            done += 1
            print(f"  {ds} {os.path.basename(path)[:44]:44s} native {native_fs:.0f} Hz done",
                  flush=True)

    df = pd.DataFrame(rows)
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RES, exist_ok=True)
    df.round(6).to_csv(os.path.join(RES, "Table_resample_methods.csv"), index=False)

    print("\nband power relative to NATIVE rate (1.000 = nothing changed):")
    piv = df.pivot_table(index="band", columns="method", values="ratio_vs_native",
                         aggfunc="median")
    piv = piv.reindex([b[0] for b in BANDS] + ["broadband_SD"])
    print(piv.round(4).to_string())
    inband = piv.drop(index=["broadband_SD"])
    print("\nworst deviation from native, per method, within 0.5-45 Hz:")
    for m in inband.columns:
        v = inband[m].dropna()
        print(f"   {m:16s} max |1 - ratio| = {np.abs(1 - v).max()*100:6.3f}%")
    print(f"\nsaved {os.path.join(RES, 'Table_resample_methods.csv')}")


if __name__ == "__main__":
    main()
