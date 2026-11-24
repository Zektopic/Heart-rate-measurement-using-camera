import argparse
import os
import sys

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import QApplication, QFileDialog

from processing import ProcessingThread
from signal_processing import METHODS, quality_label
from ui import QUALITY_COLORS, UI


def to_pixmap(frame, device_pixel_ratio=1.0):
    """BGR numpy frame (already scaled by the worker) -> QPixmap."""
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
    pixmap = QPixmap.fromImage(image)  # copies the pixels, frame can be freed
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


class App(UI):
    def __init__(self):
        super().__init__()
        self.processing_thread = ProcessingThread()
        self.processing_thread.view_signal.connect(self.update_views)
        self.video_display.installEventFilter(self)
        self.face_display.installEventFilter(self)
        self.processing_thread.bpm_signal.connect(self.update_bpm)
        self.processing_thread.plot_signal.connect(self.update_plots)
        self.processing_thread.status_signal.connect(self.statusBar().showMessage)
        self.processing_thread.finished.connect(self.on_finished)

        self._quality = None

        self.start_button.clicked.connect(self.toggle_processing)
        self.open_button.clicked.connect(self.open_video)
        self.save_button.clicked.connect(self.save_csv)
        self.input_combobox.currentIndexChanged.connect(self.set_input_source)
        self.camera_spinbox.valueChanged.connect(self.set_camera)
        self.method_combobox.currentIndexChanged.connect(self.set_method)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_processing)
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_video)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_csv)

    def toggle_processing(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
            return
        if not self.processing_thread.webcam and not self.processing_thread.video_file:
            self.open_video()  # video mode without a file: ask for one first
            return
        self.update_bpm(0.0, float("-inf"))
        self.processing_thread.start()
        self.start_button.setText("Stop")
        self._set_controls_enabled(False)

    def on_finished(self):
        """Thread ended: stopped by the user, end of video, or capture error."""
        self.start_button.setText("Start")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled):
        self.input_combobox.setEnabled(enabled)
        self.camera_spinbox.setEnabled(enabled and self.processing_thread.webcam)
        self.open_button.setEnabled(enabled)
        self.method_combobox.setEnabled(enabled)
        self.save_button.setEnabled(enabled and bool(self.processing_thread.log.rows))

    def set_camera(self, index):
        self.processing_thread.camera_index = index

    def set_method(self, index):
        self.processing_thread.method = METHODS[index]

    def save_csv(self):
        """Write the readings of the last measurement (one per second) to a CSV file."""
        if self.processing_thread.isRunning() or not self.processing_thread.log.rows:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save readings", "heart_rate.csv", "CSV files (*.csv)")
        if not filename:
            return
        try:
            self.processing_thread.log.write_csv(filename)
        except OSError as e:
            self.statusBar().showMessage("Cannot save %s: %s" % (os.path.basename(filename), e.strerror))
            return
        self.statusBar().showMessage("Saved %d readings to %s" % (len(self.processing_thread.log.rows),
                                                                  os.path.basename(filename)))

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "",
                                                  "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not filename:
            return
        self.processing_thread.set_video_file(filename)
        self.source_label.setText(os.path.basename(filename))
        self.input_combobox.setCurrentIndex(1)
        if not self.processing_thread.isRunning():
            self.toggle_processing()

    def set_input_source(self, index):
        self.processing_thread.webcam = index == 0
        self.camera_spinbox.setEnabled(index == 0)

    def update_views(self, frame, face):
        ratio = self.devicePixelRatioF()
        self.video_display.setPixmap(to_pixmap(frame, ratio))
        if face is not None:
            self.face_display.setPixmap(to_pixmap(face, ratio))
        self.processing_thread.view_shown()

    def _update_view_sizes(self):
        """Tell the worker how large to scale the frames (in device pixels)."""
        ratio = self.devicePixelRatioF()

        def device_size(label):
            return (int(label.width() * ratio), int(label.height() * ratio))

        self.processing_thread.view_sizes = (device_size(self.video_display), device_size(self.face_display))

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize and obj in (self.video_display, self.face_display):
            self._update_view_sizes()
        return super().eventFilter(obj, event)

    def update_bpm(self, bpm, snr):
        measured = bpm > 0
        self.bpm_label.setText(f"{bpm:.0f}" if measured else "--")
        quality = quality_label(snr) if measured else ""
        if quality != self._quality:  # restyle only on change
            self._quality = quality
            self.bpm_label.setProperty("quality", quality)
            self.bpm_label.style().unpolish(self.bpm_label)
            self.bpm_label.style().polish(self.bpm_label)
            self.quality_label.setStyleSheet("color: %s;" % QUALITY_COLORS.get(quality, "transparent"))
        self.quality_label.setText(f"Signal {quality} ({snr:.1f} dB)" if measured else "")
        self.fft_peak.setVisible(measured)
        self.fft_peak.setValue(bpm)

    def update_plots(self, samples, freqs, spectrum):
        self.signal_curve.setData(samples)
        if freqs is not None:
            self.fft_curve.setData(freqs, spectrum)
        self.processing_thread.plots_shown()

    def closeEvent(self, event):
        self.processing_thread.stop()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Camera based heart-rate monitor")
    parser.add_argument("--camera", type=int, default=0, metavar="N", help="camera index (default 0)")
    parser.add_argument("--video", metavar="FILE", help="measure on this video file and start at once")
    parser.add_argument("--method", choices=METHODS, default="green",
                        help="pulse signal: green channel (default) or POS (more robust to light and motion)")
    return parser.parse_known_args(argv)  # unknown arguments are left to Qt


def main(argv=None):
    args, qt_args = parse_args(argv)
    app = QApplication(sys.argv[:1] + qt_args)
    window = App()
    window.camera_spinbox.setValue(args.camera)
    window.method_combobox.setCurrentIndex(METHODS.index(args.method))
    window.show()
    if args.video:
        window.processing_thread.set_video_file(args.video)
        window.source_label.setText(os.path.basename(args.video))
        window.input_combobox.setCurrentIndex(1)
        window.toggle_processing()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
