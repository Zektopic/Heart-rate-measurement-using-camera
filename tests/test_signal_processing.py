import numpy as np
import pytest

from signal_processing import HeartRateEstimator, Signal_processing, extract_color


def feed(est, bpm, fps=30.0, seconds=12.0, noise=0.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps + rng.normal(0, jitter, n).cumsum() * 0
    t = t + rng.uniform(-jitter, jitter, n)
    t.sort()
    values = 120 + 0.8 * np.sin(2 * np.pi * bpm / 60 * t) + rng.normal(0, noise, n)
    for v, ts in zip(values, t):
        est.push(v, ts)


@pytest.mark.parametrize("bpm", [55, 72, 90, 120, 160])
def test_estimates_clean_sine(bpm):
    est = HeartRateEstimator()
    feed(est, bpm)
    assert est.ready
    assert est.bpm == pytest.approx(bpm, abs=2.5)


def test_estimates_noisy_and_jittery_signal():
    est = HeartRateEstimator()
    feed(est, 75, noise=0.3, jitter=0.004)
    assert est.bpm == pytest.approx(75, abs=4)


def test_slow_drift_is_ignored():
    est = HeartRateEstimator()
    fps = 30.0
    for i in range(400):
        t = i / fps
        est.push(100 + 0.05 * i + 0.8 * np.sin(2 * np.pi * 1.25 * t), t)
    assert est.bpm == pytest.approx(75, abs=3)


def test_window_never_exceeds_buffer_size():
    est = HeartRateEstimator(buffer_size=50)
    feed(est, 70, seconds=10)
    assert len(est._values) == 50 and len(est._times) == 50


def test_not_ready_until_window_full():
    est = HeartRateEstimator()
    assert est.progress == 0
    assert est.push(120.0, 0.0) is False
    assert est.bpm == 0.0 and not est.ready


def test_outlier_is_replaced_by_previous_value():
    est = HeartRateEstimator()
    feed(est, 72, seconds=5)
    last = est._values[-1]
    est.push(last + 500, est._times[-1] + 1 / 30)
    assert est._values[-1] == last


def test_constant_signal_gives_no_estimate_and_no_warning(recwarn):
    est = HeartRateEstimator()
    for i in range(150):
        est.push(100.0, i / 30)
    assert est.bpm == 0.0
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_zero_duration_is_rejected():
    est = HeartRateEstimator(buffer_size=10)
    for _ in range(10):
        est.push(1.0, 5.0)
    assert est.bpm == 0.0


def test_low_frame_rate_does_not_crash():
    est = HeartRateEstimator()
    feed(est, 70, fps=5.0, seconds=40)  # nyquist (2.5 Hz) < band edge (3 Hz)
    assert est.ready
    assert np.isfinite(est.samples).all()


def test_reset_clears_state():
    est = HeartRateEstimator()
    feed(est, 72)
    est.reset()
    assert est.bpm == 0.0 and est.progress == 0 and est.samples.size == 0


def test_fft_is_zero_padded_and_in_band():
    sp = Signal_processing()
    power, bpm = sp.fft(np.hamming(100) * np.sin(np.arange(100) / 30 * 2 * np.pi * 1.2), 30)
    assert bpm.min() >= 50 and bpm.max() <= 180
    assert np.diff(bpm).max() < 3  # far finer than the 18 bpm raw bin size


def test_extract_color_skips_empty_rois():
    roi = np.zeros((4, 4, 3), np.uint8)
    roi[:, :, 1] = 100
    assert extract_color([roi, roi[:0]]) == 100
    assert extract_color([roi[:0], None]) is None
    assert extract_color([]) is None
