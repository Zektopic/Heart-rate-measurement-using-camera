from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
from signal_processing import Signal_processing

class ProcessingThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    bpm_signal = pyqtSignal(float)
    plot_signal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.video_file = None
        self.webcam = True
        self.process = Process()

    def set_video_file(self, filename):
        self.video_file = filename
        self.webcam = False

    def run(self):
        self.running = True
        if self.webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.video_file)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.process.frame_in = frame
            ret_process = self.process.run()

            if ret_process:
                self.frame_signal.emit(self.process.frame_out)
                self.bpm_signal.emit(self.process.bpm)
                self.plot_signal.emit(self.process.samples, [self.process.freqs, self.process.fft])
            else:
                self.frame_signal.emit(frame)

        cap.release()
        self.running = False

    def stop(self):
        self.running = False

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()

    def run(self):
        frame = self.frame_in
        ret_process = self.fu.no_age_gender_face_process(frame, "5")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process

        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = self.sp.extract_color(ROIs)

        self.frame_out = frame
        self.frame_ROI = aligned_face
        
        L = len(self.data_buffer)
        
        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99):
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        if L == self.buffer_size:
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            self.fft = np.abs(raw)**2
        
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            idx2 = np.argmax(pruned)
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)

        self.samples = processed
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y
