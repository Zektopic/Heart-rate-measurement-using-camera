"""Face tracking (fewer detector runs) and GUI back-pressure."""
import numpy as np
import pytest

from processing import fit

dlib = pytest.importorskip("dlib")

from processing import Process, ProcessingThread  # noqa: E402
from synthetic import clip  # noqa: E402


def count_detections(process):
    calls = []
    detect = process.fu.face_detection

    def counting(*args, **kwargs):
        calls.append(1)
        return detect(*args, **kwargs)

    process.fu.face_detection = counting
    return calls


@pytest.mark.parametrize("motion", [False, True])
def test_tracking_skips_most_detections_and_keeps_accuracy(motion):
    process = Process(redetect_every=5)
    calls = count_detections(process)
    found = sum(process.run(frame, t) for frame, t in clip(84, motion=motion))
    assert found == 450
    assert len(calls) <= 450 / 5 + 10
    assert process.bpm == pytest.approx(84, abs=3)


def test_tracking_box_stays_on_the_face():
    tracked, detected = Process(redetect_every=5), Process(redetect_every=1)
    worst = 0.0
    for frame, t in clip(72, seconds=5, motion=True):
        assert tracked.run(frame, t) and detected.run(frame, t)
        a, b = tracked.rect, detected.rect
        offset = np.hypot(a.center().x - b.center().x, a.center().y - b.center().y)
        worst = max(worst, offset / b.width())
    assert worst < 0.15  # within 15 % of the face width of a fresh detection


def test_face_lost_while_tracking_is_noticed():
    process = Process(redetect_every=5)
    for frame, t in clip(72, seconds=1):
        assert process.run(frame, t)
    blank = np.full((480, 640, 3), 128, np.uint8)
    results = [process.run(blank, 1 + i / 30) for i in range(5)]
    assert results[-1] is False  # at the latest at the next detection


def test_fit_keeps_aspect_ratio():
    frame = np.zeros((480, 640, 3), np.uint8)
    assert fit(frame, (320, 400)).shape == (240, 320, 3)
    assert fit(frame, (1280, 960)).shape == (960, 1280, 3)
    assert fit(frame, None) is frame
    assert fit(None, (10, 10)) is None


def test_slow_gui_gets_the_newest_frame_instead_of_a_queue(monkeypatch):
    from PyQt6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])  # noqa: F841 - keeps Qt alive
    frames = clip(72, seconds=2)

    class Capture:
        def read(self):
            frame = next(frames, None)
            return frame is not None, None if frame is None else frame[0]

        def get(self, prop):
            return 30.0

        def release(self):
            pass

    thread = ProcessingThread()
    thread.webcam = True
    views = []
    thread.view_signal.connect(lambda frame, face: views.append(frame))
    monkeypatch.setattr(thread, "_open_capture", lambda: Capture())
    thread.run()  # synchronous; nobody acknowledges the frames
    assert len(views) == 1
    thread.view_shown()
    assert not thread.view_pending
