import os
import sys

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QShortcut

from processing import ProcessingThread
from ui import UI


def to_pixmap(frame, device_pixel_ratio=1.0):
    """BGR numpy frame (already scaled by the worker) -> QPixmap."""
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
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

        self.start_button.clicked.connect(self.toggle_processing)
        self.open_button.clicked.connect(self.open_video)
        self.input_combobox.currentIndexChanged.connect(self.set_input_source)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_processing)
        QShortcut(QKeySequence.Open, self, activated=self.open_video)

    def toggle_processing(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
            return
        if not self.processing_thread.webcam and not self.processing_thread.video_file:
            self.open_video()  # video mode without a file: ask for one first
            return
        self.bpm_label.setText("--")
        self.processing_thread.start()
        self.start_button.setText("Stop")
        self.input_combobox.setEnabled(False)
        self.open_button.setEnabled(False)

    def on_finished(self):
        """Thread ended: stopped by the user, end of video, or capture error."""
        self.start_button.setText("Start")
        self.input_combobox.setEnabled(True)
        self.open_button.setEnabled(True)

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
        if event.type() == QEvent.Resize and obj in (self.video_display, self.face_display):
            self._update_view_sizes()
        return super().eventFilter(obj, event)

    def update_bpm(self, bpm):
        self.bpm_label.setText(f"{bpm:.0f}" if bpm > 0 else "--")
        self.fft_peak.setVisible(bpm > 0)
        self.fft_peak.setValue(bpm)

    def update_plots(self, samples, freqs, spectrum):
        self.signal_curve.setData(samples)
        if freqs is not None:
            self.fft_curve.setData(freqs, spectrum)
        self.processing_thread.plots_shown()

    def closeEvent(self, event):
        self.processing_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
