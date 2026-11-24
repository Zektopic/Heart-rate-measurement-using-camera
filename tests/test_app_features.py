"""Headless video analysis and the GUI controls. Needs dlib and the models."""
import csv

import cv2
import numpy as np
import pytest

dlib = pytest.importorskip("dlib")


@pytest.fixture(scope="module")
def video_72(tmp_path_factory):
    from synthetic import clip

    path = tmp_path_factory.mktemp("video") / "face72.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for frame, _ in clip(72, seconds=14, noise=1.0):
        out.write(frame)
    out.release()
    return path


@pytest.mark.parametrize("method", ["green", "pos"])
def test_analyze_video_reports_the_pulse(video_72, tmp_path, method, capsys):
    import analyze_video

    out = tmp_path / "readings.csv"
    assert analyze_video.main([str(video_72), "--csv", str(out), "--method", method]) == 0
    rows = list(csv.DictReader(open(out, newline="")))
    assert len(rows) >= 3
    assert np.median([float(r["bpm"]) for r in rows]) == pytest.approx(72, abs=3)
    assert "median" in capsys.readouterr().out


def test_analyze_video_refuses_non_videos(tmp_path, capsys):
    import analyze_video

    fake = tmp_path / "clip.avi"
    fake.write_text("ffconcat version 1.0\nfile other.avi\n")
    assert analyze_video.main([str(fake)]) == 2
    assert "error" in capsys.readouterr().err


@pytest.fixture
def window(monkeypatch):
    from PyQt6.QtWidgets import QApplication

    import app as app_module

    qapp = QApplication.instance() or QApplication([])
    win = app_module.App()
    yield win
    win.close()
    qapp.processEvents()


def test_gui_controls_reach_the_worker(window):
    window.method_combobox.setCurrentIndex(1)
    window.camera_spinbox.setValue(2)
    assert window.processing_thread.method == "pos"
    assert window.processing_thread.camera_index == 2
    window.input_combobox.setCurrentIndex(1)
    assert not window.camera_spinbox.isEnabled()


def test_gui_shows_quality(window):
    window.update_bpm(72.4, 8.0)
    assert window.bpm_label.text() == "72"
    assert window.quality_label.text() == "Signal good (8.0 dB)"
    assert window.bpm_label.property("quality") == "good"
    window.update_bpm(0.0, float("-inf"))
    assert window.bpm_label.text() == "--" and window.quality_label.text() == ""


def test_gui_saves_csv(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    target = tmp_path / "hr.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    window.processing_thread.log.add(0.0, 70.0, 5.0)
    window.on_finished()
    assert window.save_button.isEnabled()
    window.save_csv()
    assert target.read_text().splitlines()[1] == "0.0,70.0,5.0,fair"


def test_command_line_options():
    import app as app_module

    args, rest = app_module.parse_args(["--camera", "1", "--method", "pos", "--video", "a.mp4", "-style", "fusion"])
    assert (args.camera, args.method, args.video) == (1, "pos", "a.mp4")
    assert rest == ["-style", "fusion"]
