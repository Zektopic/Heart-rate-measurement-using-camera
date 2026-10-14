import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QShortcut

from processing import ProcessingThread
from ui import UI


def to_pixmap(frame, size):
    """BGR numpy frame -> QPixmap scaled (keeping aspect ratio) to fit `size`."""
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(image).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class App(UI):
    def __init__(self):
        super().__init__()
        self.processing_thread = ProcessingThread()
        self.processing_thread.frame_signal.connect(self.update_video_display)
        self.processing_thread.face_signal.connect(self.update_face_display)
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

    def update_video_display(self, frame):
        self.video_display.setPixmap(to_pixmap(frame, self.video_display.size()))

    def update_face_display(self, face):
        self.face_display.setPixmap(to_pixmap(face, self.face_display.size()))

    def update_bpm(self, bpm):
        self.bpm_label.setText(f"{bpm:.0f}" if bpm > 0 else "--")
        self.fft_peak.setVisible(bpm > 0)
        self.fft_peak.setValue(bpm)

    def update_plots(self, samples, freqs, spectrum):
        self.signal_curve.setData(samples)
        if freqs is not None:
            self.fft_curve.setData(freqs, spectrum)

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
