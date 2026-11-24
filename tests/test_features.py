"""Time based window, POS, signal quality and the CSV log."""
import csv

import numpy as np
import pytest

from signal_processing import HeartRateEstimator, pos_pulse, quality_label, spectral_snr

SKIN_BGR = np.array([100.0, 140.0, 180.0])
PULSE_BGR = np.array([0.53, 1.0, 0.33])  # pulse strength per channel, green strongest


def run_estimator(est, bpm, fps=30.0, seconds=20.0, noise=0.0, flicker=0.0, amplitude=0.3, seed=0):
    """Feed (B, G, R) or green samples; returns the absolute errors once ready."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    pulse = amplitude * np.sin(2 * np.pi * bpm / 60 * t)
    light = 1 + flicker * np.sin(2 * np.pi * 1.5 * t)  # in-band brightness change (90 bpm)
    errors = []
    for i in range(n):
        bgr = (SKIN_BGR + pulse[i] * PULSE_BGR) * light[i] + rng.normal(0, noise, 3)
        est.push(bgr if est.method == "pos" else bgr[1], t[i])
        if est.ready:
            errors.append(abs(est.bpm - bpm))
    return np.array(errors)


@pytest.mark.parametrize("fps", [10.0, 30.0, 60.0])
def test_time_window_is_ready_after_window_seconds_at_any_frame_rate(fps):
    est = HeartRateEstimator(window_seconds=10)
    n = int(9.5 * fps)
    for i in range(n):
        est.push(120 + np.sin(i / fps * 2 * np.pi * 1.2), i / fps)
    assert not est.ready and 0.9 < est.progress < 1.0
    for i in range(n, n + int(fps)):
        est.push(120 + np.sin(i / fps * 2 * np.pi * 1.2), i / fps)
    assert est.ready and est.progress == 1.0
    assert abs(len(est._values) - 10 * fps) <= 1
    assert est.bpm == pytest.approx(72, abs=2)


def test_time_window_is_more_accurate_than_100_samples_at_30_fps():
    short = np.mean([run_estimator(HeartRateEstimator(buffer_size=100), b, noise=0.6, seed=s).mean()
                     for b in (60, 75, 100, 120) for s in range(3)])
    long = np.mean([run_estimator(HeartRateEstimator(window_seconds=10), b, noise=0.6, seed=s).mean()
                    for b in (60, 75, 100, 120) for s in range(3)])
    assert long < 2 and short > 3 * long


def test_pos_ignores_in_band_lighting_flicker_that_fools_green():
    green = run_estimator(HeartRateEstimator(window_seconds=10), 66, noise=0.05, flicker=0.01)
    pos = run_estimator(HeartRateEstimator(window_seconds=10, method="pos"), 66, noise=0.05, flicker=0.01)
    assert green.mean() > 15  # locked on the 90 bpm flicker
    assert pos.mean() < 2


@pytest.mark.parametrize("bpm", [55, 72, 110, 150])
def test_pos_measures_clean_pulse(bpm):
    errors = run_estimator(HeartRateEstimator(window_seconds=10, method="pos"), bpm, noise=0.05)
    assert errors.mean() < 2


def test_pos_pulse_is_zero_for_constant_colour_and_handles_short_input():
    assert np.allclose(pos_pulse(np.tile(SKIN_BGR, (60, 1)), 30.0), 0)
    assert pos_pulse(np.tile(SKIN_BGR, (5, 1)), 30.0).shape == (5,)


def test_method_must_be_known_and_pos_needs_three_channels():
    with pytest.raises(ValueError):
        HeartRateEstimator(method="ica")
    with pytest.raises(ValueError):
        HeartRateEstimator(method="pos").push(120.0, 0.0)


def test_snr_separates_pulse_from_noise():
    clean = HeartRateEstimator(window_seconds=10)
    run_estimator(clean, 72, noise=0.05)
    noise_only = HeartRateEstimator(window_seconds=10)
    run_estimator(noise_only, 72, noise=1.0, amplitude=0.0)
    assert clean.quality == "good" and clean.snr > 10
    assert noise_only.quality == "poor" and noise_only.snr < 2


def test_snr_and_quality_labels():
    bpm = np.linspace(50, 180, 261)
    peak = np.exp(-((bpm - 72) / 3) ** 2)
    assert spectral_snr(bpm, peak + 1e-4, 72, 30.0, 300) > 10
    assert spectral_snr(bpm, np.ones_like(bpm), 72, 30.0, 300) < 0
    assert [quality_label(x) for x in (8, 3, -1, float("-inf"))] == ["good", "fair", "poor", "poor"]


def test_measurement_log_keeps_one_row_per_second(tmp_path):
    from processing import MeasurementLog

    log = MeasurementLog()
    for i in range(90):
        log.add(10 + i / 30, 72.04, 7.123)
    assert [row[0] for row in log.rows] == [10.0, 11.0, 12.0]
    log.write_csv(tmp_path / "out.csv")
    rows = list(csv.reader(open(tmp_path / "out.csv", newline="")))
    assert rows[0] == ["time_s", "bpm", "snr_db", "quality"]
    assert rows[1] == ["10.0", "72.0", "7.12", "good"]
