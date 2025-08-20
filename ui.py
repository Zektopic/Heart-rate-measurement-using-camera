from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QLabel, QPushButton, QComboBox, QGridLayout, QWidget
import pyqtgraph as pg

class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Monitor")
        self.setGeometry(100, 100, 1200, 600)
        self.center()
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2e2e2e;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;
            }
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
            QComboBox {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 10px;
                font-size: 16px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        self.video_display = QLabel()
        self.video_display.setStyleSheet("background-color: #000000")
        self.layout.addWidget(self.video_display, 0, 0, 2, 1)

        self.face_display = QLabel()
        self.face_display.setStyleSheet("background-color: #000000")
        self.layout.addWidget(self.face_display, 0, 1)

        self.signal_plot = pg.PlotWidget()
        self.layout.addWidget(self.signal_plot, 1, 1)

        self.fft_plot = pg.PlotWidget()
        self.layout.addWidget(self.fft_plot, 1, 2)

        self.bpm_label = QLabel("BPM: 0")
        self.layout.addWidget(self.bpm_label, 0, 2)

        self.start_button = QPushButton("Start")
        self.layout.addWidget(self.start_button, 2, 0)

        self.input_combobox = QComboBox()
        self.input_combobox.addItems(["Webcam", "Video"])
        self.layout.addWidget(self.input_combobox, 2, 1)
        
        self.open_button = QPushButton("Open Video")
        self.layout.addWidget(self.open_button, 2, 2)


        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
