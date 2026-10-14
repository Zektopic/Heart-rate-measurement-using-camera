import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from face_utilities import Face_utilities
from signal_processing import HeartRateEstimator, extract_color


class Process(object):
    """Frame -> heart rate pipeline. Qt free, so it can be used and tested on its own."""

    def __init__(self, landmarks="5", buffer_size=100):
        self.landmarks = landmarks
        self.fu = Face_utilities()
        self.estimator = HeartRateEstimator(buffer_size=buffer_size)
        self.frame_out = None
        self.frame_ROI = None
        self.rect = None

    # read-only views of the estimator state
    bpm = property(lambda self: self.estimator.bpm)
    fps = property(lambda self: self.estimator.fps)
    samples = property(lambda self: self.estimator.samples)
    freqs = property(lambda self: self.estimator.freqs)
    fft = property(lambda self: self.estimator.spectrum)

    def reset(self):
        self.estimator.reset()
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
        green = extract_color(rois)
        if green is None:
            return False

        self.estimator.push(green, timestamp)
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
    frame_signal = pyqtSignal(np.ndarray)           # camera frame (BGR)
    face_signal = pyqtSignal(np.ndarray)            # aligned face (BGR)
    bpm_signal = pyqtSignal(float)
    plot_signal = pyqtSignal(object, object, object)  # pulse signal, bpm axis, spectrum
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.video_file = None
        self.webcam = True
        self.process = Process()

    def set_video_file(self, filename):
        self.video_file = filename
        self.webcam = False

    def _open_capture(self):
        if self.webcam:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, no lag
        else:
            cap = cv2.VideoCapture(self.video_file)
        return cap

    def run(self):
        self.running = True
        self.process.reset()
        cap = self._open_capture()
        if not cap.isOpened():
            self.status_signal.emit("Cannot open " + ("the camera" if self.webcam else "the video file"))
            cap.release()
            self.running = False
            return

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
                self.frame_signal.emit(self.process.draw_overlay(frame))
                self.face_signal.emit(self.process.frame_ROI)
                if est.ready:
                    self.bpm_signal.emit(est.bpm)
                    self.plot_signal.emit(est.samples, est.freqs, est.spectrum)
                    self.status_signal.emit("Measuring - %.0f fps" % shown_fps)
                else:
                    self.plot_signal.emit(est.samples, None, None)
                    self.status_signal.emit("Collecting data... %d%%" % (est.progress * 100))
            else:
                self.frame_signal.emit(frame)
                self.status_signal.emit("No face detected")
                # a long gap would corrupt the window: start over when the face is back
                if timestamp - last_face > 1.5 and self.process.estimator.progress > 0:
                    self.process.reset()
                    self.bpm_signal.emit(0.0)

        cap.release()
        self.running = False

    def stop(self):
        self.running = False
        self.wait(3000)
