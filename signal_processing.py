"""Signal processing for camera based (remote PPG) heart-rate estimation.

The pipeline is:  colour mean of the cheek ROIs  ->  rolling window (10 s by
default)  ->  resample to an even time grid  ->  pulse signal (green channel,
or POS from all three channels)  ->  detrend  ->  Hamming window  ->
zero-padded FFT  ->  strongest peak between 50 and 180 bpm, with its SNR.
"""
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

MIN_BPM = 50.0
MAX_BPM = 180.0
BAND_LOW_HZ = 0.8   # 48 bpm
BAND_HIGH_HZ = 3.0  # 180 bpm
MAX_WINDOW_SAMPLES = 4096  # cap of a time based window (about 2 minutes at 30 fps)
METHODS = ("green", "pos")

# signal quality from the SNR of the spectral peak (dB)
QUALITY_GOOD_DB = 6.0
QUALITY_FAIR_DB = 2.0


def extract_color(rois):
    """Mean green value over all non-empty ROIs, or None if there is no pixel."""
    bgr = extract_bgr(rois)
    return None if bgr is None else float(bgr[1])


def extract_bgr(rois):
    """Mean (blue, green, red) over all non-empty ROIs, or None if there is no pixel."""
    means = [roi.reshape(-1, roi.shape[-1])[:, :3].mean(axis=0) for roi in rois if roi is not None and roi.size]
    if not means:
        return None
    return np.mean(means, axis=0)


def pos_pulse(bgr, fps, window_seconds=1.6):
    """Pulse signal from (n, 3) blue, green, red means with POS.

    Plane-Orthogonal-to-Skin, W. Wang et al., "Algorithmic Principles of
    Remote PPG", IEEE TBME 2017: in short overlapping windows the colour is
    normalised by its mean and projected on a plane orthogonal to the skin
    tone, which cancels intensity changes (motion, lighting) common to all
    channels; the windows are overlap-added.
    """
    rgb = np.asarray(bgr, float)[:, ::-1]
    n = len(rgb)
    length = min(max(int(round(window_seconds * fps)), 2), n)
    windows = sliding_window_view(rgb, length, axis=0).transpose(0, 2, 1)  # (n - length + 1, length, 3)
    mean = windows.mean(axis=1, keepdims=True)
    normalised = windows / np.where(mean > 0, mean, 1.0)
    s1 = normalised[..., 1] - normalised[..., 2]                                   # G - B
    s2 = -2 * normalised[..., 0] + normalised[..., 1] + normalised[..., 2]         # -2R + G + B
    std1, std2 = s1.std(axis=1, keepdims=True), s2.std(axis=1, keepdims=True)
    h = s1 + np.divide(std1, std2, out=np.zeros_like(std1), where=std2 > 0) * s2
    h -= h.mean(axis=1, keepdims=True)
    pulse = np.zeros(n)
    for k in range(length):  # overlap-add
        pulse[k:k + len(h)] += h[:, k]
    return pulse


def spectral_snr(bpm_axis, power, peak_bpm, fps, n):
    """SNR (dB) of the spectral peak and its first harmonic against the rest of the band.

    After de Haan and Jeanne, "Robust pulse rate from chrominance-based rPPG",
    IEEE TBME 2013, with the peak width set to the Hamming main lobe of an
    `n` sample window (two FFT bins) instead of a fixed 0.1 Hz.
    """
    width = max(6.0, 2 * 60.0 * fps / n)
    in_peak = (np.abs(bpm_axis - peak_bpm) <= width) | (np.abs(bpm_axis - 2 * peak_bpm) <= width)
    signal_power, noise_power = power[in_peak].sum(), power[~in_peak].sum()
    if noise_power <= 0:
        return float("inf") if signal_power > 0 else float("-inf")
    return float(10 * np.log10(max(signal_power, 1e-300) / noise_power))


def quality_label(snr_db):
    """'good', 'fair' or 'poor' for an SNR in dB."""
    if snr_db >= QUALITY_GOOD_DB:
        return "good"
    if snr_db >= QUALITY_FAIR_DB:
        return "fair"
    return "poor"


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
    ``bpm`` (median of the last ``smoothing`` raw estimates), ``snr`` (signal
    quality of the latest spectrum, dB), ``samples`` (the band-passed pulse
    signal) and ``freqs`` / ``spectrum`` (for plotting).

    Args:
        buffer_size: window length in samples (used when window_seconds is None)
        window_seconds: window length in seconds. Better than a sample count:
            the frequency resolution and the noise rejection then do not
            depend on the frame rate (100 samples are 12 s at 8 fps but only
            3.3 s at 30 fps)
        method: "green" (value = green mean) or "pos" (value = (B, G, R) means)
    """

    def __init__(self, buffer_size=100, outlier_threshold=10.0, smoothing=5, window_seconds=None,
                 method="green"):
        if method not in METHODS:
            raise ValueError("method must be one of %s" % (METHODS,))
        self.window_seconds = window_seconds
        self.buffer_size = buffer_size if window_seconds is None else MAX_WINDOW_SAMPLES
        self.outlier_threshold = outlier_threshold
        self.method = method
        self._values = deque(maxlen=self.buffer_size)
        self._times = deque(maxlen=self.buffer_size)
        self._estimates = deque(maxlen=smoothing)
        self._sos_cache = {}
        self._sp = Signal_processing()
        self.reset()

    def reset(self):
        self._values.clear()
        self._times.clear()
        self._estimates.clear()
        self._full = False
        self.bpm = 0.0
        self.snr = float("-inf")
        self.fps = 0.0
        self.samples = np.zeros(0)
        self.freqs = np.zeros(0)
        self.spectrum = np.zeros(0)

    @property
    def quality(self):
        """'good', 'fair' or 'poor' for the latest estimate."""
        return quality_label(self.snr)

    @property
    def progress(self):
        """Fraction (0..1) of the window that has been filled."""
        if self._full:
            return 1.0
        if self.window_seconds is None:
            return len(self._values) / self.buffer_size
        if len(self._times) < 2:
            return 0.0
        return min((self._times[-1] - self._times[0]) / self.window_seconds, 1.0)

    @property
    def ready(self):
        if self.window_seconds is None:
            return len(self._values) == self.buffer_size
        return self._full

    def _green(self, value):
        return value if np.ndim(value) == 0 else value[1]

    def push(self, value, timestamp):
        """Add one sample (a green mean, or (B, G, R) means for POS).

        Returns True if a new estimate was computed.
        """
        if self.method == "pos":
            value = np.asarray(value, float)
            if value.shape != (3,):
                raise ValueError("the POS method needs (B, G, R) means")
        elif np.ndim(value):
            value = float(value[1])  # (B, G, R) given: green method uses green
        if self._values:
            green_mean = np.mean([self._green(v) for v in self._values]) if self.method == "pos" \
                else np.mean(self._values)
            if self.ready and abs(self._green(value) - green_mean) > self.outlier_threshold:
                value = self._values[-1]  # reject sudden jumps (motion, lighting)
        self._values.append(value)
        self._times.append(timestamp)
        if self.window_seconds is not None:
            while self._times[-1] - self._times[0] > self.window_seconds:
                self._times.popleft()
                self._values.popleft()
                self._full = True
            if len(self._values) == self.buffer_size:
                self._full = True

        if not self.ready:
            if self.method == "green":
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
        n = len(self._values)
        times = np.fromiter(self._times, float, n)
        values = np.asarray(self._values, float)

        duration = times[-1] - times[0]
        if duration <= 0 or n < 8:
            return False
        self.fps = (n - 1) / duration

        # resample onto an even time grid, build the pulse signal, detrend and window
        even = np.linspace(times[0], times[-1], n)
        if values.ndim == 1:
            pulse = np.interp(even, times, values)
        else:
            pulse = pos_pulse(np.column_stack([np.interp(even, times, values[:, c]) for c in range(3)]), self.fps)
        resampled = signal.detrend(pulse)
        if np.ptp(resampled) < 1e-6 * max(np.abs(pulse).max(), 1e-3):  # flat signal: only float noise
            return False
        windowed = np.hamming(n) * resampled
        norm = np.linalg.norm(windowed)

        power, bpm_axis = self._sp.fft(windowed / norm, self.fps)
        if power.size == 0:
            return False
        self.freqs, self.spectrum = bpm_axis, power
        peak = float(bpm_axis[np.argmax(power)])
        self.snr = spectral_snr(bpm_axis, power, peak, self.fps, n)
        self._estimates.append(peak)
        self.bpm = float(np.median(self._estimates))
        self.samples = self._bandpass(resampled, self.fps)
        return True
