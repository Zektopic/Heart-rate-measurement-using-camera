"""The stand-alone demo scripts: graph_plot.py and amplify_color.py."""
import numpy as np
import pytest

import amplify_color
import graph_plot
from synthetic import clip


def test_graph_plot_reads_signal_file():
    red, green, blue = graph_plot.load_signal()
    assert len(red) == len(green) == len(blue) > 500
    bpm, power = graph_plot.spectrum(green)
    band = (bpm >= 50) & (bpm <= 180)
    assert np.isfinite(power).all()
    assert 50 <= bpm[band][np.argmax(power[band])] <= 180


def test_ideal_filter_keeps_band_when_low_edge_is_dc():
    t = np.arange(40) / 30.0
    tone = np.sin(2 * np.pi * 1.5 * t)[:, None]
    out = amplify_color.VidMag.temporal_ideal_filter(tone, 0.0, 3.0, 30.0)
    assert np.abs(out).max() > 0.5  # fft[-0:] = 0 used to wipe everything


@pytest.mark.parametrize("motion", [False, True])
def test_vidmag_outputs_newest_frame(motion):
    vm = amplify_color.VidMag(buffer_size=40)
    out = None
    for frame, _ in clip(72, seconds=1.5, size=(256, 192)):
        out = vm.push(frame.astype(np.float32), motion=motion)
    assert out is not None
    assert out.shape == (192, 256, 3)
    assert np.isfinite(out).all()


def test_amplify_color_window_runs_until_the_camera_ends(monkeypatch):
    frames = [f for f, _ in clip(72, seconds=2, size=(320, 240))]

    class FakeCamera:
        def __init__(self, index):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    monkeypatch.setattr(amplify_color.cv2, "VideoCapture", FakeCamera)
    vm = amplify_color.VidMag(buffer_size=20)
    assert vm.main_loop() == 0
    assert vm.frame_out is not None and vm.frame_out.shape[1] == 640
