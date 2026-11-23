# Heart-rate-measurement-using-camera
[![Alt text](https://github.com/habom2310/Heart-rate-measurement-using-camera/blob/master/result.JPG)](https://youtu.be/JeQGSEXk_BQ)
# Abstract
- Heart Rate (HR) is one of the most important Physiological parameter and a vital indicator of people‘s physiological state
- A non-contact based system to measure Heart Rate: real-time application using camera
- Principal: extract heart rate information from facial skin color variation caused by blood circulation 
- Application: monitoring drivers‘ physiological state

# Methods 
- Detect face, align and get ROI using facial landmarks (the face is tracked through its landmarks between detections)
- Apply band pass filter with fl = 0.8 Hz and fh = 3 Hz, which are 48 and 180 bpm respectively
- The average colour of the ROIs in each frame is pushed to a rolling window of the last 10 seconds
- The pulse signal is the green channel (**Green**, default) or the POS projection of all three channels
  (**POS**, W. Wang et al. 2017), which cancels brightness changes caused by lighting and motion
- Resample to an even time grid, detrend, apply a Hamming window and take the zero-padded FFT. The highest peak between 50 and 180 bpm is the heart rate (the median of the last 5 estimates is shown)
- Signal quality is the SNR of that peak (and its harmonic) against the rest of the band: good from 6 dB, fair from
  2 dB, poor below. A poor reading is shown greyed out. A periodic disturbance (e.g. flickering light) can still give a
  high SNR at the wrong rate; POS helps against that
- Amplify color to make the color variation visible 

# Requirements
```
pip install -r requirements.txt
```


# Implementation
```
python app.py [--camera N] [--video FILE] [--method green|pos]
```
- Pick **Webcam** (and the camera number) or **Video file** (Open video... / Ctrl+O), choose **Green** or **POS**,
  then press **Start** (or Space) to start and stop the measurement
- Video files are analysed against their own timestamps and played back in real time
- **Save CSV...** (Ctrl+S) writes the readings of the last measurement, one per second: time, bpm, SNR, quality
- Without the GUI, as fast as the computer allows:
  ```
  python analyze_video.py VIDEO [--csv readings.csv] [--method green|pos] [--window 10]
  ```
- The `.dat` landmark models are loaded from the folder of the code, so the app can be started from any directory
- `python graph_plot.py` plots the spectrum of the recorded `signal.dat` (needs `pip install matplotlib`)
- `python amplify_color.py [--camera N] [--motion]` is the Eulerian Video Magnification demo (Esc quits)

# Security
- Video files are untrusted input. They are opened through `video_source.py`: local regular files only, and FFmpeg
  limited to the `file` protocol and ordinary container formats (mp4/mov, avi, mkv/webm, mpeg, wmv, flv). Playlists
  (`.m3u8`), ffconcat scripts and other formats that make FFmpeg open further files or URLs are refused, whatever
  their file name.
- The landmark models are checked against the SHA-256 of the official dlib files before dlib parses them.
- An error while processing a frame ends the measurement with a message instead of aborting the app.
- CI runs `pip-audit` on `requirements.txt`; Dependabot proposes updates for the pinned packages and Actions.

# Tests
```
pip install pytest
python -m pytest
```
The end-to-end test runs a synthetic clip (the face from `result.JPG` with a pulse added) through the whole pipeline.
CI runs `ruff check .` and the tests on Python 3.12 and 3.13.

# Results
- Data from a specialized device, Compact 5 medical Econet, is used for the ground truth. In certain circumstances, the Heart rate values measured using the application and the device are the same

# Reference
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
- Graphs plotting is based on https://github.com/thearn/webcam-pulse-detector
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# Note
- Application can only detect HR for 1 people at a time
- Sudden change can cause incorrect HR calculation. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera
- This github project is for study purpose only. For other purposes, please contact me at khanhhanguyen2310@gmail.com

