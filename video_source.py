"""Opening cameras and video files.

Video files are untrusted input. OpenCV hands them to FFmpeg, which picks a
demuxer by probing the file, and some demuxers open further inputs named in
the file: an ffconcat script plays the files it lists even when it is saved
as ".avi" (the pinned opencv-python does this), and an HLS ".m3u8" playlist
fetches its URLs when FFmpeg was built with network support. Files are
therefore opened with FFmpeg limited to the local file protocol and to
ordinary container formats, whatever the file is called and however
OpenCV was built.
"""
import os
from pathlib import Path

import cv2

# FFmpeg demuxer names of the containers the app plays (mp4/mov, avi, mkv/webm,
# mpeg-ps/ts, wmv, flv). Not included: hls, dash, concat, image2, tee and the like.
ALLOWED_DEMUXERS = "mov,mp4,m4a,3gp,3g2,mj2,avi,matroska,webm,mpeg,mpegts,asf,flv"
FFMPEG_FILE_OPTIONS = "protocol_whitelist;file|format_whitelist;" + ALLOWED_DEMUXERS

MAX_FRAME_SIDE = 8192  # refuse absurd frame sizes (decompression bombs)


class VideoSourceError(ValueError):
    """The requested video source cannot or must not be opened."""


def check_video_path(path):
    """Return the resolved path of an existing regular file, or raise VideoSourceError."""
    text = os.fspath(path)
    if "://" in text:
        raise VideoSourceError("Only local video files can be opened")
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_file():  # also rejects devices, FIFOs and directories
        raise VideoSourceError("Not a video file: %s" % resolved.name)
    return resolved


def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(int(index))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, no lag
    return cap


def open_video_file(path):
    """Open a local video file through FFmpeg with the restrictions above.

    Raises VideoSourceError when the path is not a local regular file, when
    FFmpeg refuses or cannot read it, or when the frame size is absurd.
    """
    resolved = check_video_path(path)
    # read by OpenCV's FFmpeg backend on every open
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_FILE_OPTIONS
    cap = cv2.VideoCapture(str(resolved), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise VideoSourceError("Cannot open the video file %s" % resolved.name)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if w > MAX_FRAME_SIDE or h > MAX_FRAME_SIDE:
        cap.release()
        raise VideoSourceError("Video frames are too large (%dx%d)" % (w, h))
    return cap
