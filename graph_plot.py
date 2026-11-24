"""Plot the spectrum of the recorded RGB signal in signal.dat (needs matplotlib).

Each line of signal.dat is "red%green%blue", recorded at 30 fps.
"""
from pathlib import Path

import numpy as np
from scipy import signal

FS = 30.0
SIGNAL_FILE = Path(__file__).resolve().parent / "signal.dat"


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    sos = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data)


def load_signal(path=SIGNAL_FILE):
    """(red, green, blue) arrays from a "r%g%b" per line file."""
    rgb = np.loadtxt(path, delimiter="%", ndmin=2)
    return rgb[:, 0], rgb[:, 1], rgb[:, 2]


def spectrum(values, fs=FS):
    """Band-passed, windowed power spectrum: (bpm axis, power)."""
    filtered = butter_bandpass_filter(signal.detrend(values), 0.8, 3, fs)
    windowed = np.hamming(len(filtered)) * filtered
    power = np.abs(np.fft.rfft(windowed / np.linalg.norm(windowed))) ** 2
    bpm = np.fft.rfftfreq(len(filtered), 1 / fs) * 60
    return bpm, power


def main():
    import matplotlib.pyplot as plt

    _, green, _ = load_signal()
    bpm, power = spectrum(green)
    band = (bpm >= 50) & (bpm <= 180)
    peak = bpm[band][np.argmax(power[band])]

    fig = plt.figure("green")
    ax = fig.add_subplot(111)
    ax.set_title("Green channel spectrum (peak %.1f bpm)" % peak)
    ax.set_xlabel("rate (bpm)")
    ax.set_ylabel("power")
    ax.plot(bpm[band], power[band], color="green")
    plt.show()


if __name__ == "__main__":
    main()
