import csv
import time
import traceback

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from face_utilities import Face_utilities
from signal_processing import HeartRateEstimator, extract_bgr, quality_label
from video_source import VideoSourceError, open_camera, open_video_file


def fit(image, size):
    """Scale `image` (keeping its aspect ratio) to fit `size` = (width, height).

    Big reductions first halve the image with a Gaussian pyramid, then
    interpolate linearly: as smooth as INTER_AREA and Qt's smooth scaling but
    about 4x cheaper on 1080p frames.
    """
    if image is None or size is None:
        return image
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    if scale <= 0 or abs(scale - 1) < 0.01:
        return image
    new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
    while image.shape[1] >= 2 * new_size[0] and image.shape[0] >= 2 * new_size[1]:
        image = cv2.pyrDown(image)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


class MeasurementLog:
    """One row per `interval` seconds of measurement: time, bpm, SNR, quality."""

    HEADER = ("time_s", "bpm", "snr_db", "quality")

    def __init__(self, interval=1.0):
        self.interval = interval
        self.rows = []

    def clear(self):
        self.rows = []

    def add(self, timestamp, bpm, snr):
        if self.rows and timestamp - self.rows[-1][0] < self.interval:
            return False
        self.rows.append((round(float(timestamp), 3), round(float(bpm), 1), round(float(snr), 2), quality_label(snr)))
        return True

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            writer.writerows(self.rows)


class Process(object):
    """Frame -> heart rate pipeline. Qt free, so it can be used and tested on its own.

    redetect_every: the face detector runs at most every this many frames;
    in between the face is tracked through its landmarks (see Face_utilities).
    window_seconds: length of the analysis window. buffer_size: a window of
    this many samples instead (the old behaviour).
    method: "green" or "pos", see signal_processing.
    """

    def __init__(self, landmarks="5", buffer_size=None, redetect_every=5, window_seconds=10.0, method="green"):
        self.landmarks = landmarks
        self.method = method
        self.fu = Face_utilities(redetect_every=redetect_every)
        if buffer_size is not None:
            self.estimator = HeartRateEstimator(buffer_size=buffer_size, method=method)
        else:
            self.estimator = HeartRateEstimator(window_seconds=window_seconds, method=method)
        self.frame_out = None
        self.frame_ROI = None
        self.rect = None

    # read-only views of the estimator state
    bpm = property(lambda self: self.estimator.bpm)
    snr = property(lambda self: self.estimator.snr)
    fps = property(lambda self: self.estimator.fps)
    samples = property(lambda self: self.estimator.samples)
    freqs = property(lambda self: self.estimator.freqs)
    fft = property(lambda self: self.estimator.spectrum)

    def reset(self):
        self.estimator.reset()
        self.fu.reset_tracking()
        self.frame_out = None
        self.frame_ROI = None
        self.rect = None

    def run(self, frame, timestamp):
        """Process one BGR frame taken at `timestamp` (seconds).

        Returns False when no usable face was found in the frame.
        """
        found = self.fu.no_age_gender_face_process(frame, self.landmarks)
        if found is None:
            return False
        rects, _, _, aligned_face, aligned_shape = found

        rois = self.fu.ROI_extraction(aligned_face, aligned_shape)
        bgr = extract_bgr(rois)
        if bgr is None:
            return False

        self.estimator.push(bgr if self.method == "pos" else float(bgr[1]), timestamp)
        self.frame_out = frame
        self.frame_ROI = aligned_face
        self.rect = rects[0]
        return True

    def draw_overlay(self, frame):
        """Face rectangle on the camera frame (in place)."""
        r = self.rect
        cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), (0, 220, 0), 2)
        return frame


class ProcessingThread(QThread):
    """Runs the measurement off the GUI thread.

    Frames and plots are only sent when the GUI has shown the previous ones
    (call view_shown() / plots_shown() from the slots). A slow GUI then skips
    frames instead of queueing them, which kept growing the delay and memory.
    """

    view_signal = pyqtSignal(object, object)          # camera frame, aligned face (BGR, scaled to the views)
    bpm_signal = pyqtSignal(float, float)             # bpm (0: none), SNR in dB
    plot_signal = pyqtSignal(object, object, object)  # pulse signal, bpm axis, spectrum
    status_signal = pyqtSignal(str)

    STATUS_INTERVAL = 0.5  # seconds between fps updates in the status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        # ((w, h) of the camera view, (w, h) of the face view) in device pixels,
        # set by the GUI so frames are scaled here and not on the GUI thread
        self.view_sizes = (None, None)
        self.view_pending = False
        self.plot_pending = False
        self._status = None
        self._status_time = 0.0
        self.video_file = None
        self.webcam = True
        self.camera_index = 0
        self.method = "green"
        self.process = Process(method=self.method)
        self.log = MeasurementLog()

    def view_shown(self):
        self.view_pending = False

    def plots_shown(self):
        self.plot_pending = False

    def _send_view(self, frame, face):
        if self.view_pending:
            return
        self.view_pending = True
        frame_size, face_size = self.view_sizes
        self.view_signal.emit(fit(frame, frame_size), fit(face, face_size))

    def _send_plots(self, samples, freqs, spectrum):
        if self.plot_pending:
            return
        self.plot_pending = True
        self.plot_signal.emit(samples, freqs, spectrum)

    def _send_status(self, text, throttle=False):
        now = time.monotonic()
        if text == self._status or (throttle and now - self._status_time < self.STATUS_INTERVAL):
            return
        self._status, self._status_time = text, now
        self.status_signal.emit(text)

    def set_video_file(self, filename):
        self.video_file = filename
        self.webcam = False

    def _open_capture(self):
        """Returns an opened capture, or None after reporting the error."""
        if self.webcam:
            cap = open_camera(self.camera_index)
            if cap.isOpened():
                return cap
            cap.release()
            self.status_signal.emit("Cannot open camera %d" % self.camera_index)
            return None
        try:
            return open_video_file(self.video_file)
        except VideoSourceError as e:
            self.status_signal.emit(str(e))
            return None

    def run(self):
        # An exception escaping QThread.run() makes PyQt abort the whole
        # application, so a bad frame or file ends the measurement instead.
        self.running = True
        self.view_pending = self.plot_pending = False
        self._status = None
        cap = None
        try:
            if self.process.method != self.method:
                self.process = Process(method=self.method)
            self.process.reset()
            self.log.clear()
            cap = self._open_capture()
            if cap is not None:
                self._measure(cap)
        except Exception as e:  # reported to the user instead, see above
            traceback.print_exc()
            self.status_signal.emit("Error: %s" % e)
        finally:
            if cap is not None:
                cap.release()
            self.running = False

    def _measure(self, cap):
        # A video file is processed against its own clock (frame index / fps),
        # not wall time, or the heart rate would depend on how fast we decode.
        video_fps = 0.0 if self.webcam else cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        start = time.monotonic()
        index = 0
        last_tick = start
        frame_time = None  # moving average of the processing time per frame
        last_face = 0.0    # timestamp of the last frame with a face

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.webcam:
                timestamp = time.monotonic() - start
            else:
                timestamp = index / video_fps
                # play back in real time instead of racing through the file
                ahead = timestamp - (time.monotonic() - start)
                if ahead > 0:
                    self.msleep(int(ahead * 1000))
            index += 1

            found = self.process.run(frame, timestamp)
            now = time.monotonic()
            dt = max(now - last_tick, 1e-3)
            last_tick = now
            frame_time = dt if frame_time is None else 0.9 * frame_time + 0.1 * dt
            shown_fps = 1.0 / frame_time

            if found:
                last_face = timestamp
                est = self.process.estimator
                if not self.view_pending:
                    self._send_view(self.process.draw_overlay(frame), self.process.frame_ROI)
                if est.ready:
                    self.bpm_signal.emit(est.bpm, est.snr)
                    self.log.add(timestamp, est.bpm, est.snr)
                    self._send_plots(est.samples, est.freqs, est.spectrum)
                    self._send_status("Measuring - %.0f fps" % shown_fps, throttle=True)
                else:
                    self._send_plots(est.samples, None, None)
                    self._send_status("Collecting data... %d%%" % (est.progress * 100))
            else:
                self._send_view(frame, None)
                self._send_status("No face detected")
                # a long gap would corrupt the window: start over when the face is back
                if timestamp - last_face > 1.5 and self.process.estimator.progress > 0:
                    self.process.reset()
                    self.bpm_signal.emit(0.0, float("-inf"))

    def stop(self):
        self.running = False
        self.wait(3000)
