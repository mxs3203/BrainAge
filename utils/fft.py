import pandas as pd
from scipy.signal import get_window, welch


def compute_fft_bands(segments_by_task, config, fs=256, nperseg=128, noverlap=64):
    fft_bands = config["fft_bands"]
    fft_segments = {}

    # Identify EEG columns automatically
    example_key = next(iter(segments_by_task))
    if not segments_by_task[example_key]:
        raise ValueError("No segments found in dictionary.")
    example_df = segments_by_task[example_key][0]
    eeg_cols = [c for c in example_df.columns if c.startswith("EEG.")]

    for task, seg_list in segments_by_task.items():
        fft_segments[task] = []
        for seg_df in seg_list:
            n_samples = len(seg_df)
            seg_nperseg = min(nperseg, n_samples)
            seg_noverlap = min(noverlap, seg_nperseg // 2)
            window = get_window("hamming", seg_nperseg)

            #
            freqs, psd = welch(
                seg_df[eeg_cols].values,
                fs=fs,
                window=window,
                nperseg=seg_nperseg,
                noverlap=seg_noverlap,
                axis=0,
                detrend="constant"
            )

            psd_df = pd.DataFrame(psd, index=freqs, columns=eeg_cols)

            # Compute mean band power
            band_results = {}
            for band_name, (fmin, fmax) in fft_bands.items():
                mask = (freqs >= fmin) & (freqs < fmax)
                if not mask.any():
                    continue  # skip empty bands
                band_power = psd_df.loc[mask].mean(axis=0)
                for ch in eeg_cols:
                    band_results[f"{ch}_{band_name}"] = band_power[ch]

            band_df = pd.DataFrame([band_results])
            fft_segments[task].append(band_df)

    return fft_segments
