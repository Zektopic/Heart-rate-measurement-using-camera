"""End to end: synthetic face clip -> Process -> heart rate. Needs dlib and the models."""
import pytest

pytest.importorskip("dlib")

from processing import Process  # noqa: E402
from synthetic import clip  # noqa: E402


@pytest.mark.parametrize("bpm, motion", [(72, False), (96, False), (84, True)])
def test_process_measures_synthetic_pulse(bpm, motion):
    process = Process()
    found = sum(process.run(frame, t) for frame, t in clip(bpm, motion=motion))
    assert found >= 440  # the detector may miss a frame or two while the head turns
    assert process.estimator.ready
    assert process.bpm == pytest.approx(bpm, abs=3)


def test_process_reports_no_face():
    import numpy as np

    process = Process()
    assert process.run(np.zeros((480, 640, 3), np.uint8), 0.0) is False
    assert process.estimator.progress == 0
