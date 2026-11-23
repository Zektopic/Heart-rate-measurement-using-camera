import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

BG = "#1b1d23"
PANEL = "#252932"
BORDER = "#343a46"
TEXT = "#e6e8ec"
MUTED = "#8b93a1"
ACCENT = "#ff4d5a"
QUALITY_COLORS = {"good": "#3ecf8e", "fair": "#f5b942", "poor": "#8b93a1"}

STYLE = f"""
    QMainWindow, QWidget#root {{ background-color: {BG}; }}
    QLabel {{ color: {TEXT}; font-size: 14px; }}
    QLabel#caption {{ color: {MUTED}; font-size: 12px; font-weight: 600; letter-spacing: 1px; }}
    QLabel#bpm {{ color: {ACCENT}; font-size: 64px; font-weight: 700; }}
    QLabel#bpmUnit {{ color: {MUTED}; font-size: 16px; }}
    QLabel#bpm[quality="poor"] {{ color: {MUTED}; }}
    QLabel#quality {{ font-size: 13px; font-weight: 600; }}
    QLabel#view {{ background-color: #000000; border: 1px solid {BORDER}; border-radius: 6px; }}
    QFrame#card {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
    QFrame#card QLabel {{ background: transparent; }}
    QPushButton, QComboBox, QSpinBox {{
        background-color: {PANEL}; color: {TEXT}; border: 1px solid {BORDER};
        border-radius: 6px; padding: 8px 16px; font-size: 14px; min-height: 20px;
    }}
    QPushButton:hover, QComboBox:hover {{ background-color: #2f3440; }}
    QPushButton:pressed {{ background-color: #20242c; }}
    QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: #5c6370; }}
    QPushButton#primary {{ background-color: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
    QPushButton#primary:hover {{ background-color: #ff6772; }}
    QComboBox QAbstractItemView {{
        background-color: {PANEL}; color: {TEXT}; selection-background-color: #3a4150;
    }}
    QStatusBar {{ color: {MUTED}; }}
"""


def _card(title, content):
    """A titled panel around `content`."""
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    caption = QLabel(title.upper())
    caption.setObjectName("caption")
    layout.addWidget(caption)
    layout.addWidget(content, 1)
    return card


class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Monitor")
        self.resize(1200, 700)
        self.setMinimumSize(900, 560)
        self.center()
        self.setStyleSheet(STYLE)

        pg.setConfigOptions(antialias=False, background=PANEL, foreground=MUTED)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        grid = QGridLayout(root)
        grid.setContentsMargins(16, 16, 16, 12)
        grid.setSpacing(12)

        # camera / video view; Ignored lets the window shrink below the pixmap size
        self.video_display = self._view("Start the measurement to see the camera")
        grid.addWidget(self.video_display, 0, 0, 3, 1)

        # BPM readout
        self.bpm_label = QLabel("--")
        self.bpm_label.setObjectName("bpm")
        self.bpm_label.setAlignment(Qt.AlignCenter)
        unit = QLabel("beats per minute")
        unit.setObjectName("bpmUnit")
        unit.setAlignment(Qt.AlignCenter)
        bpm_box = QWidget()
        bpm_layout = QVBoxLayout(bpm_box)
        bpm_layout.setContentsMargins(0, 0, 0, 0)
        bpm_layout.setSpacing(0)
        bpm_layout.addWidget(self.bpm_label)
        bpm_layout.addWidget(unit)
        self.quality_label = QLabel("")
        self.quality_label.setObjectName("quality")
        self.quality_label.setAlignment(Qt.AlignCenter)
        self.quality_label.setToolTip("Signal quality: SNR of the heart-rate peak against the rest of the spectrum")
        bpm_layout.addWidget(self.quality_label)

        # aligned face (what the cheek ROIs are cut from)
        self.face_display = self._view("")
        self.face_display.setMinimumSize(120, 120)

        side = QWidget()
        side_layout = QHBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        side_layout.addWidget(_card("Heart rate", bpm_box), 3)
        side_layout.addWidget(_card("Aligned face", self.face_display), 2)
        grid.addWidget(side, 0, 1)

        self.signal_plot = self._plot("Time (frames)")
        self.signal_curve = self.signal_plot.plot(pen=pg.mkPen(ACCENT, width=2))
        grid.addWidget(_card("Pulse signal", self.signal_plot), 1, 1)

        self.fft_plot = self._plot("Rate (bpm)")
        self.fft_plot.setXRange(50, 180, padding=0)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen("#4da3ff", width=2))
        self.fft_peak = pg.InfiniteLine(angle=90, pen=pg.mkPen(ACCENT, style=Qt.DashLine))
        self.fft_peak.setVisible(False)
        self.fft_plot.addItem(self.fft_peak)
        grid.addWidget(_card("Spectrum", self.fft_plot), 2, 1)

        # controls
        controls = QHBoxLayout()
        controls.setSpacing(12)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumWidth(120)
        self.start_button.setToolTip("Start / stop (Space)")
        self.input_combobox = QComboBox()
        self.input_combobox.addItems(["Webcam", "Video file"])
        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setRange(0, 9)
        self.camera_spinbox.setPrefix("Camera ")
        self.camera_spinbox.setToolTip("Camera index (0 is the default camera)")
        self.open_button = QPushButton("Open video...")
        self.open_button.setToolTip("Ctrl+O")
        self.method_combobox = QComboBox()
        self.method_combobox.addItems(["Green", "POS"])
        self.method_combobox.setToolTip("Green: green channel only (best when still).\n"
                                        "POS: all three channels, cancels lighting and motion changes.")
        self.save_button = QPushButton("Save CSV...")
        self.save_button.setToolTip("Save the readings of the last measurement (Ctrl+S)")
        self.save_button.setEnabled(False)
        self.source_label = QLabel("")
        self.source_label.setObjectName("caption")
        self.source_label.setTextFormat(Qt.PlainText)  # file names are not markup
        controls.addWidget(self.start_button)
        controls.addWidget(self.input_combobox)
        controls.addWidget(self.camera_spinbox)
        controls.addWidget(self.open_button)
        controls.addWidget(self.method_combobox)
        controls.addWidget(self.source_label, 1)
        controls.addWidget(self.save_button)
        grid.addLayout(controls, 3, 0, 1, 2)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 2)
        grid.setRowStretch(2, 2)

        self.statusBar().showMessage("Ready")

    @staticmethod
    def _view(text):
        label = QLabel(text)
        label.setObjectName("view")
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setMinimumSize(320, 240)
        return label

    @staticmethod
    def _plot(x_label):
        plot = pg.PlotWidget()
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(False, False)
        plot.hideButtons()
        plot.getPlotItem().hideAxis("left")
        plot.setLabel("bottom", x_label)
        plot.showGrid(x=True, y=False, alpha=0.2)
        return plot

    def center(self):
        geometry = self.frameGeometry()
        geometry.moveCenter(self.screen().availableGeometry().center())
        self.move(geometry.topLeft())
