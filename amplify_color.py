"""Eulerian Video Magnification (colour) demo: makes the pulse visible.

    python amplify_color.py [--camera N] [--motion]

Shows the camera and the amplified view; Esc quits.
"""
import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import scipy.signal as signal


class VidMag:
    """Colour (or motion) magnification over a rolling window of frames.

    Only the newest frame is reconstructed: the window keeps the small
    pyramid level of each frame instead of the full frames, so the cost per
    frame is one pyramid, one small FFT and one reconstruction.
    """

    def __init__(self, buffer_size=40, width=256, levels=3):
        self.buffer_size = buffer_size
        self.width = width
        self.levels = levels
        self.fps = 0.0
        self.times = deque(maxlen=buffer_size)
        self.data_buffer = deque(maxlen=buffer_size)  # gaussian (or laplacian) level per frame
        self.frame_out = None

    # -------------------- COLOUR MAGNIFICATION -------------------- #
    @staticmethod
    def build_gaussian_pyramid(src, level=3):
        s = src.copy()
        pyramid = [s]
        for _ in range(level):
            s = cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    @staticmethod
    def temporal_ideal_filter(tensor, low, high, fps, axis=0):
        fft = np.fft.fft(tensor, axis=axis)
        frequencies = np.fft.fftfreq(tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        if bound_low:  # fft[-0:] would clear the whole spectrum
            fft[-bound_low:] = 0
        # the real part keeps the sign of the variation (abs() rectified it)
        return np.fft.ifft(fft, axis=axis).real

    def reconstruct_frame(self, amplified, origin):
        img = amplified
        for _ in range(self.levels):
            img = cv2.pyrUp(img)
        img = cv2.resize(img, (origin.shape[1], origin.shape[0]))
        return img + origin

    def magnify_color(self, low=0.4, high=2.0, amplification=30):
        tensor = np.asarray(self.data_buffer, dtype=np.float32)
        filtered = self.temporal_ideal_filter(tensor, low, high, self.fps)
        return filtered[-1] * amplification

    # -------------------- MOTION MAGNIFICATION -------------------- #
    def build_laplacian_pyramid(self, src, levels=3):
        gaussian = self.build_gaussian_pyramid(src, levels)
        pyramid = []
        for i in range(levels, 0, -1):
            up = cv2.pyrUp(gaussian[i], dstsize=(gaussian[i - 1].shape[1], gaussian[i - 1].shape[0]))
            pyramid.append(cv2.subtract(gaussian[i - 1], up))
        return pyramid

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        sos = signal.butter(order, [lowcut / nyq, min(highcut / nyq, 0.99)], btype="band", output="sos")
        return signal.sosfilt(sos, data, axis=0)

    def magnify_motion(self, low=0.4, high=1.5, amplification=30):
        """Band-pass every laplacian level over time; returns the newest amplified detail."""
        levels = list(zip(*self.data_buffer, strict=True))  # per level: list of frames
        up = None
        for level in levels:
            filtered = self.butter_bandpass_filter(np.asarray(level, np.float32), low, high, self.fps)[-1]
            filtered *= amplification
            up = filtered if up is None else cv2.pyrUp(up, dstsize=(filtered.shape[1], filtered.shape[0])) + filtered
        return up

    # ---------------------------------------------------------------- #
    def push(self, small, motion=False):
        now = time.monotonic()
        self.times.append(now)
        if motion:
            self.data_buffer.append(self.build_laplacian_pyramid(small, self.levels))
        else:
            self.data_buffer.append(self.build_gaussian_pyramid(small, self.levels)[-1])
        if len(self.data_buffer) < self.buffer_size:
            return None
        self.fps = (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)
        if motion:
            return small + self.magnify_motion()
        return self.reconstruct_frame(self.magnify_color(), small)

    def process_frame(self, frame, motion=False):
        """Camera frame -> (frame with the FPS drawn on it, newest amplified frame or None)."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(h * self.width / w))).astype(np.float32)
        out = self.push(small, motion)
        if out is not None:
            amplified = cv2.convertScaleAbs(out)
            self.frame_out = cv2.resize(amplified, (640, int(amplified.shape[0] * 640 / amplified.shape[1])))
        cv2.putText(frame, "FPS %.2f" % self.fps, (20, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        return frame, self.frame_out

    def main_loop(self, camera=0, motion=False):
        # OpenCV is the headless build (no bundled Qt 5), so the window is PyQt6
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QKeySequence, QShortcut
        from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget

        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print("[ERROR] cannot open the camera", file=sys.stderr)
            return 1
        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = QWidget()
        window.setWindowTitle("Original | Colour amplification (Esc quits)")
        layout = QHBoxLayout(window)
        views = (QLabel("Starting the camera..."), QLabel("Collecting frames..."))
        for view in views:
            layout.addWidget(view)
        QShortcut(QKeySequence("Esc"), window, activated=window.close)

        def step():
            ok, frame = cap.read()
            if not ok:
                window.close()
                return
            for view, image in zip(views, self.process_frame(frame, motion), strict=True):
                if image is not None:
                    view.setPixmap(to_pixmap(image))

        timer = QTimer(window)
        timer.timeout.connect(step)
        timer.start(0)
        window.show()
        app.exec()
        timer.stop()
        cap.release()
        return 0


def to_pixmap(frame):
    """BGR uint8 numpy frame -> QPixmap."""
    from PyQt6.QtGui import QImage, QPixmap

    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    return QPixmap.fromImage(QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--motion", action="store_true", help="magnify motion instead of colour")
    args = parser.parse_args(argv)
    return VidMag().main_loop(camera=args.camera, motion=args.motion)


if __name__ == "__main__":
    sys.exit(main())
