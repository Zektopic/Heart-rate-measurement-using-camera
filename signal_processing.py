"""Signal processing for camera based (remote PPG) heart-rate estimation.

The pipeline is:  green-channel mean of the cheek ROIs  ->  rolling buffer
->  resample to an even time grid  ->  detrend  ->  Hamming window  ->
zero-padded FFT  ->  strongest peak between 50 and 180 bpm.
"""
from collections import deque

import numpy as np
from scipy import signal

MIN_BPM = 50.0
MAX_BPM = 180.0
BAND_LOW_HZ = 0.8   # 48 bpm
BAND_HIGH_HZ = 3.0  # 180 bpm


def extract_color(rois):
    """Mean green value over all non-empty ROIs, or None if there is no pixel."""
    means = [float(roi[:, :, 1].mean()) for roi in rois if roi is not None and roi.size]
    if not means:
        return None
    return float(np.mean(means))


def butter_bandpass_sos(lowcut, highcut, fs, order=3):
    """Band-pass design as second-order sections (numerically stable).

    Returns None when the sampling rate is too low for the requested band.
    """
    nyq = 0.5 * fs
    if nyq <= lowcut * 1.05:
        return None
    high = min(highcut, nyq * 0.99)
    return signal.butter(order, [lowcut / nyq, high / nyq], btype="band", output="sos")


class Signal_processing:
    """Stateless helpers kept under their original names for compatibility."""

    def extract_color(self, ROIs):
        return extract_color(ROIs)

    def normalization(self, data_buffer):
        return data_buffer / np.linalg.norm(data_buffer)

    def signal_detrending(self, data_buffer):
        return signal.detrend(data_buffer)

    def interpolation(self, data_buffer, times):
        """Resample to an even time grid and apply a Hamming window."""
        n = len(data_buffer)
        even_times = np.linspace(times[0], times[-1], n)
        return np.hamming(n) * np.interp(even_times, times, data_buffer)

    def fft(self, data_buffer, fps, pad_factor=8):
        """Power spectrum restricted to the heart-rate band.

        Returns (power, bpm_axis).
        """
        n = len(data_buffer)
        n_fft = n * pad_factor
        bpm_axis = 60.0 * np.fft.rfftfreq(n_fft, d=1.0 / fps)
        power = np.abs(np.fft.rfft(data_buffer, n=n_fft)) ** 2
        keep = (bpm_axis >= MIN_BPM) & (bpm_axis <= MAX_BPM)
        return power[keep], bpm_axis[keep]

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=3):
        sos = butter_bandpass_sos(lowcut, highcut, fs, order)
        if sos is None:
            return signal.detrend(data_buffer)
        return signal.sosfiltfilt(sos, data_buffer)


class HeartRateEstimator:
    """Rolling-window heart-rate estimator.

    Feed it one (value, timestamp) pair per frame with :meth:`push`. Once the
    window is full every push produces a fresh estimate, available as
    ``bpm`` (median of the last ``smoothing`` raw estimates), ``samples`` (the
    band-passed pulse signal) and ``freqs`` / ``spectrum`` (for plotting).
    """

    def __init__(self, buffer_size=100, outlier_threshold=10.0, smoothing=5):
        self.buffer_size = buffer_size
        self.outlier_threshold = outlier_threshold
        self._values = deque(maxlen=buffer_size)
        self._times = deque(maxlen=buffer_size)
        self._estimates = deque(maxlen=smoothing)
        self._sos_cache = {}
        self._sp = Signal_processing()
        self.reset()

    def reset(self):
        self._values.clear()
        self._times.clear()
        self._estimates.clear()
        self.bpm = 0.0
        self.fps = 0.0
        self.samples = np.zeros(0)
        self.freqs = np.zeros(0)
        self.spectrum = np.zeros(0)

    @property
    def progress(self):
        """Fraction (0..1) of the window that has been filled."""
        return len(self._values) / self.buffer_size

    @property
    def ready(self):
        return len(self._values) == self.buffer_size

    def push(self, value, timestamp):
        """Add one sample. Returns True if a new estimate was computed."""
        if self.ready and abs(value - np.mean(self._values)) > self.outlier_threshold:
            value = self._values[-1]  # reject sudden jumps (motion, lighting)
        self._values.append(value)
        self._times.append(timestamp)

        if not self.ready:
            self.samples = signal.detrend(np.fromiter(self._values, float, len(self._values)))
            return False
        return self._estimate()

    def _bandpass(self, data, fps):
        key = round(fps * 2)  # cache designs per 0.5 fps step
        if key not in self._sos_cache:
            self._sos_cache[key] = butter_bandpass_sos(BAND_LOW_HZ, BAND_HIGH_HZ, key / 2)
        sos = self._sos_cache[key]
        if sos is None:
            return data
        return signal.sosfiltfilt(sos, data)

    def _estimate(self):
        n = self.buffer_size
        times = np.fromiter(self._times, float, n)
        values = np.fromiter(self._values, float, n)

        duration = times[-1] - times[0]
        if duration <= 0:
            return False
        self.fps = (n - 1) / duration

        # resample onto an even time grid, then detrend and window
        even = np.linspace(times[0], times[-1], n)
        resampled = signal.detrend(np.interp(even, times, values))
        if np.ptp(resampled) < 1e-6:  # flat signal: nothing to measure, only float noise
            return False
        windowed = np.hamming(n) * resampled
        norm = np.linalg.norm(windowed)

        power, bpm_axis = self._sp.fft(windowed / norm, self.fps)
        if power.size == 0:
            return False
        self.freqs, self.spectrum = bpm_axis, power
        self._estimates.append(float(bpm_axis[np.argmax(power)]))
        self.bpm = float(np.median(self._estimates))
        self.samples = self._bandpass(resampled, self.fps)
        return True
