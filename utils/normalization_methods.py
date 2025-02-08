
from scipy.signal import detrend

"""
    Perform Z-score normalization by subtracting the mean of the baseline 
    and dividing by its standard deviation.
    Formula:
        X' = (X - mean_baseline) / std_baseline
    Args:
        signal_df (DataFrame): EEG signal data to be normalized.
        baseline_df (DataFrame): Baseline data used for normalization.
    Returns:
        DataFrame: Z-score normalized signal.
"""
def z_score_normalization(signal_df, baseline_df):
    baseline_mean = baseline_df.mean()
    baseline_std = baseline_df.std()

    normalized_stimulus = (signal_df - baseline_mean) / baseline_std
    return normalized_stimulus


"""
   Perform basic normalization by subtracting the mean of the baseline 
   from the signal.
   Formula:
       X' = X - mean_baseline
   Args:
       signal_df (DataFrame): EEG signal data to be normalized.
       baseline_df (DataFrame): Baseline data used for normalization.
   Returns:
       DataFrame: Baseline-subtracted signal.
"""
def basic_normalization(signal_df, baseline_df):
    baseline_mean = baseline_df.mean()

    normalized_stimulus = signal_df - baseline_mean
    return normalized_stimulus


"""
 Perform ratio normalization by dividing the signal by the mean of the baseline.
 Formula:
     X' = X / mean_baseline
 Argss:
     signal_df (DataFrame): EEG signal data to be normalized.
     baseline_df (DataFrame): Baseline data used for normalization.
 Returns:
     DataFrame: Ratio-normalized signal.
"""
def ratio_normalization(signal_df, baseline_df):
    baseline_mean = baseline_df.mean()

    normalized_stimulus = signal_df / baseline_mean

    return normalized_stimulus


"""
    Perform detrending to remove low-frequency drifts from the EEG signal.
    Uses SciPy's detrend function to subtract the best-fit linear trend 
    from each channel in the signal.
    Args:
        signal_df (DataFrame): EEG signal data to be detrended.
    Returns:
        DataFrame: Detrended signal.
"""
def detrend_normalization(signal_df):
    normalized_stimulus = signal_df.apply(detrend, axis=0)

    return normalized_stimulus


"""
  Perform robust normalization using the Interquartile Range (IQR).
  Formula:
      X' = (X - median_baseline) / IQR_baseline
  where:
      IQR_baseline = Q3_baseline - Q1_baseline
  Args:
      signal_df (DataFrame): EEG signal data to be normalized.
      baseline_df (DataFrame): Baseline data used for normalization.
  Returns:
      DataFrame: IQR-normalized signal.
"""
def IQR_normalization(signal_df, baseline_df):
    median_baseline = baseline_df.median()
    iqr_baseline = baseline_df.quantile(0.75) - baseline_df.quantile(0.25)

    # Normalize stimulus using robust scaling
    normalized_stimulus = (signal_df - median_baseline) / iqr_baseline
    return normalized_stimulus