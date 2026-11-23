"""Measure the heart rate in a video file without the GUI.

    python analyze_video.py VIDEO [--csv out.csv] [--method green|pos] [--window 10]

The video is processed as fast as possible against its own timestamps and
one reading per second is printed (and written to the CSV file). Video files
are opened with the same restrictions as in the app (see video_source.py).
"""
import argparse
import sys

import cv2
import numpy as np

from processing import MeasurementLog, Process
from signal_processing import METHODS
from video_source import VideoSourceError, open_video_file


def analyze(path, method="green", window_seconds=10.0, landmarks="5", progress=None):
    """Run the pipeline over a video file; returns (MeasurementLog, stats dict)."""
    cap = open_video_file(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        process = Process(landmarks=landmarks, window_seconds=window_seconds, method=method)
        log = MeasurementLog()
        frames = faces = 0
        last_face = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = msec / 1000.0 if msec and msec > 0 else frames / fps
            frames += 1
            if process.run(frame, timestamp):
                faces += 1
                last_face = timestamp
                est = process.estimator
                if est.ready and log.add(timestamp, est.bpm, est.snr) and progress:
                    progress(log.rows[-1])
            elif timestamp - last_face > 1.5 and process.estimator.progress > 0:
                process.reset()  # as in the app: a long gap would corrupt the window
    finally:
        cap.release()
    bpms = np.array([row[1] for row in log.rows])
    stats = {
        "frames": frames,
        "face_frames": faces,
        "readings": len(log.rows),
        "median_bpm": float(np.median(bpms)) if bpms.size else None,
    }
    return log, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="Heart rate from a video file, without the GUI")
    parser.add_argument("video", help="video file (mp4, avi, mov, mkv, ...)")
    parser.add_argument("--csv", metavar="FILE", help="write the readings to this CSV file")
    parser.add_argument("--method", choices=METHODS, default="green", help="pulse signal (default green)")
    parser.add_argument("--window", type=float, default=10.0, metavar="SECONDS",
                        help="analysis window (default 10 s)")
    parser.add_argument("--landmarks", choices=("5", "68"), default="5", help="dlib landmark model (default 5)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)
    if args.window < 2:
        parser.error("--window must be at least 2 seconds")

    def show(row):
        print("%8.1f s  %5.1f bpm  SNR %5.1f dB  %s" % row, flush=True)

    try:
        log, stats = analyze(args.video, args.method, args.window, args.landmarks,
                             progress=None if args.quiet else show)
    except VideoSourceError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if args.csv:
        log.write_csv(args.csv)
    median = "%.1f bpm" % stats["median_bpm"] if stats["median_bpm"] is not None else "no reading"
    print("%d frames, face in %d, %d readings, median %s" % (stats["frames"], stats["face_frames"],
                                                             stats["readings"], median))
    return 0 if stats["readings"] else 1


if __name__ == "__main__":
    sys.exit(main())
