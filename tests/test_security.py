"""Untrusted input handling: video files, model files, worker errors."""
import http.server
import os
import threading

import cv2
import numpy as np
import pytest

import video_source
from face_utilities import MODEL_SHA256, BASE_DIR, verify_model
from video_source import VideoSourceError, check_video_path, open_video_file


def write_video(path, size=(64, 48), frames=5, fourcc="MJPG"):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 30, size)
    assert out.isOpened()
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    out.release()
    return path


@pytest.fixture
def http_server():
    """Local HTTP server that records every request it gets."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], hits
    server.shutdown()


@pytest.mark.parametrize("name, fourcc", [("clip.avi", "MJPG"), ("clip.mp4", "mp4v"), ("clip.mkv", "MJPG")])
def test_ordinary_videos_still_open(tmp_path, name, fourcc):
    cap = open_video_file(write_video(tmp_path / name, fourcc=fourcc))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (48, 64, 3)


@pytest.mark.parametrize("name", ["holiday.mp4", "playlist.m3u8"])
def test_hls_playlist_is_refused(tmp_path, http_server, name):
    port, hits = http_server
    fake = tmp_path / name
    fake.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:0\n"
                    "#EXTINF:10.0,\nhttp://127.0.0.1:%d/segment.ts\n#EXT-X-ENDLIST\n" % port)
    with pytest.raises(VideoSourceError):
        open_video_file(fake)
    assert hits == []  # FFmpeg never contacted the URL named in the file


def test_ffconcat_script_disguised_as_avi_is_refused(tmp_path):
    write_video(tmp_path / "other.avi")
    fake = tmp_path / "clip.avi"
    fake.write_text("ffconcat version 1.0\nfile other.avi\n")
    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    plain = cv2.VideoCapture(str(fake))
    plain_reads = plain.isOpened() and plain.read()[0]
    plain.release()
    with pytest.raises(VideoSourceError):
        open_video_file(fake)
    if not plain_reads:
        pytest.skip("this OpenCV build does not play ffconcat scripts anyway")


@pytest.mark.parametrize("bad", ["http://127.0.0.1/a.mp4", "file:///etc/hostname", "concat:a.mp4|b.mp4"])
def test_urls_and_protocols_are_refused(bad):
    with pytest.raises(VideoSourceError):
        check_video_path(bad)


def test_non_files_are_refused(tmp_path):
    with pytest.raises(VideoSourceError):
        check_video_path(tmp_path)  # directory
    with pytest.raises(VideoSourceError):
        check_video_path(tmp_path / "missing.mp4")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "pipe.mp4")
        with pytest.raises(VideoSourceError):
            check_video_path(tmp_path / "pipe.mp4")


def test_absurd_frame_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(video_source, "MAX_FRAME_SIDE", 100)
    with pytest.raises(VideoSourceError, match="too large"):
        open_video_file(write_video(tmp_path / "wide.avi", size=(128, 16), frames=1))


def test_shipped_models_match_their_checksums():
    for kind, sha in MODEL_SHA256.items():
        verify_model(BASE_DIR / ("shape_predictor_%s_face_landmarks.dat" % kind), sha)


def test_tampered_or_missing_model_is_refused(tmp_path):
    fake = tmp_path / "shape_predictor_5_face_landmarks.dat"
    fake.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_model(fake, MODEL_SHA256["5"])
    with pytest.raises(RuntimeError, match="Cannot read"):
        verify_model(tmp_path / "missing.dat", MODEL_SHA256["5"])


def test_worker_error_ends_measurement_instead_of_aborting(monkeypatch):
    from PyQt6.QtCore import QCoreApplication

    from processing import ProcessingThread

    app = QCoreApplication.instance() or QCoreApplication([])  # noqa: F841 - keeps Qt alive

    class Capture:
        released = False

        def read(self):
            return True, np.zeros((48, 64, 3), np.uint8)

        def get(self, prop):
            return 30.0

        def release(self):
            Capture.released = True

    thread = ProcessingThread()
    messages = []
    thread.status_signal.connect(messages.append)
    monkeypatch.setattr(thread, "_open_capture", lambda: Capture())

    def boom(frame, timestamp):
        raise ValueError("corrupt frame")

    monkeypatch.setattr(thread.process, "run", boom)
    thread.run()  # synchronously: must return, not raise
    assert messages[-1] == "Error: corrupt frame"
    assert Capture.released and not thread.running
