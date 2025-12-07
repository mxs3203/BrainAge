import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks, welch
from scipy.stats import skew, kurtosis, entropy
from scipy.special import entr
from scipy.linalg import hankel
from numpy.lib.stride_tricks import sliding_window_view

class EEGFeatureExtractor:
    def __init__(self, data: pd.DataFrame):
        #assert data.shape == (512, 14), "Data must be 512x14"
        self.data = data
        self.features = {}

    @staticmethod
    def mean(x):
        return np.mean(x)

    @staticmethod
    def median(x):
        return np.median(x)

    @staticmethod
    def variance(x):
        return np.var(x)

    @staticmethod
    def std_dev(x):
        return np.std(x)

    @staticmethod
    def signal_skewness(x):
        return skew(x)

    @staticmethod
    def signal_kurtosis(x):
        return kurtosis(x)

    @staticmethod
    def peak_to_peak(x):
        return np.ptp(x)

    @staticmethod
    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    @staticmethod
    def energy(x):
        return np.sum(np.square(x))

    @staticmethod
    def zero_crossing_rate(x):
        return np.mean(np.diff(np.signbit(x)))

    @staticmethod
    def waveform_length(x):
        return np.sum(np.abs(np.diff(x)))

    @staticmethod
    def hjorth_parameters(x):
        first_deriv = np.diff(x)
        second_deriv = np.diff(first_deriv)
        var_zero = np.var(x)
        var_d1 = np.var(first_deriv)
        var_d2 = np.var(second_deriv)
        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / mobility
        return activity, mobility, complexity

    @staticmethod
    def teager_energy(x):
        return np.mean(x[1:-1]**2 - x[:-2]*x[2:])

    @staticmethod
    def approximate_entropy(x, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(x)
        N = len(x)
        def _phi(m):
            x_m = sliding_window_view(x, window_shape=m)
            C = np.sum(np.max(np.abs(x_m[:, None] - x_m[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
            return np.sum(np.log(C)) / (N - m + 1)
        return abs(_phi(m) - _phi(m + 1))

    @staticmethod
    def sample_entropy(x, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(x)
        def _phi(m):
            x_m = sliding_window_view(x, window_shape=m)
            C = np.sum(np.max(np.abs(x_m[:, None] - x_m[None, :]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (len(x_m) * (len(x_m) - 1))
        try:
            return -np.log(_phi(m+1) / _phi(m))
        except ZeroDivisionError:
            return np.inf

    @staticmethod
    def permutation_entropy(x, m=3, delay=1):
        n = len(x) - delay * (m - 1)
        permutations = np.array([np.argsort(x[i:i + delay * m:delay]) for i in range(n)])
        _, counts = np.unique(permutations, axis=0, return_counts=True)
        p = counts / counts.sum()
        return entropy(p)

    def compute(self):
        features = []
        for ch in self.data.columns:
            x = self.data[ch].values
            act, mob, comp = self.hjorth_parameters(x)
            ch_feats = [
                self.mean(x),
                self.median(x),
                self.variance(x),
                self.std_dev(x),
                self.signal_skewness(x),
                self.signal_kurtosis(x),
                self.peak_to_peak(x),
                self.rms(x),
                self.energy(x),
                self.zero_crossing_rate(x),
                self.waveform_length(x),
                act, mob, comp,
                self.teager_energy(x),
                self.approximate_entropy(x),
                self.sample_entropy(x),
                self.permutation_entropy(x)
            ]
            features.append(ch_feats)

        columns = [
            'Mean', 'Median', 'Variance', 'StdDev', 'Skewness', 'Kurtosis',
            'PeakToPeak', 'RMS', 'Energy', 'ZeroCrossRate', 'WaveformLength',
            'Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity',
            'TeagerEnergy', 'ApproxEntropy', 'SampleEntropy', 'PermutationEntropy'
        ]
        return pd.DataFrame(features, index=self.data.columns, columns=columns)
