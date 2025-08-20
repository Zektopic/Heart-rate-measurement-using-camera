import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ui import UI
from processing import ProcessingThread
import numpy as np
import cv2

class App(UI):
    def __init__(self):
        super().__init__()
        self.processing_thread = ProcessingThread()
        self.processing_thread.frame_signal.connect(self.update_video_display)
        self.processing_thread.bpm_signal.connect(self.update_bpm)
        self.processing_thread.plot_signal.connect(self.update_plots)

        self.start_button.clicked.connect(self.toggle_processing)
        self.open_button.clicked.connect(self.open_video)
        self.input_combobox.currentIndexChanged.connect(self.set_input_source)

    def toggle_processing(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.start_button.setText("Start")
        else:
            self.processing_thread.start()
            self.start_button.setText("Stop")

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.processing_thread.set_video_file(filename)
            self.input_combobox.setCurrentIndex(1)

    def set_input_source(self, index):
        if index == 0:
            self.processing_thread.webcam = True
        else:
            self.processing_thread.webcam = False

    def update_video_display(self, frame):
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888).rgbSwapped()
        self.video_display.setPixmap(QPixmap.fromImage(image))

    def update_bpm(self, bpm):
        self.bpm_label.setText(f"BPM: {bpm:.2f}")

    def update_plots(self, signal, fft):
        self.signal_plot.clear()
        self.signal_plot.plot(signal)
        self.fft_plot.clear()
        self.fft_plot.plot(fft[0], fft[1])

    def closeEvent(self, event):
        self.processing_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = App()
    sys.exit(app.exec_())
