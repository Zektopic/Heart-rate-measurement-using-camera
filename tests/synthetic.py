"""Synthetic test clips: the face from result.JPG with a pulse added to the skin."""
from pathlib import Path

import cv2
import numpy as np

RESULT_IMAGE = Path(__file__).resolve().parent.parent / "result.JPG"
CAMERA_VIEW = (slice(416, 850), slice(62, 641))  # webcam frame inside the screenshot


def face_image(size=(640, 480)):
    image = cv2.imread(str(RESULT_IMAGE))
    return cv2.resize(image[CAMERA_VIEW], size, interpolation=cv2.INTER_AREA)


def skin_mask(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (60, 135, 85), (255, 180, 135))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (15, 15), 0)[..., None]


def clip(bpm, seconds=15.0, fps=30.0, size=(640, 480), amplitude=1.2, noise=2.0, motion=False, seed=0):
    """Yield (frame, timestamp). The pulse is strongest in green, like real skin."""
    base = face_image(size).astype(np.float32)
    mask = skin_mask(base.astype(np.uint8))
    weights = np.array([0.53, 1.0, 0.33], np.float32)  # B, G, R
    rng = np.random.default_rng(seed)
    w, h = size
    for i in range(int(seconds * fps)):
        t = i / fps
        pulse = np.sin(2 * np.pi * bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * bpm / 60 * t + 0.6)
        frame = base + mask * (amplitude * pulse) * weights
        if motion:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), 6 * np.sin(2 * np.pi * 0.09 * t),
                                        1 + 0.06 * np.sin(2 * np.pi * 0.05 * t))
            m[:, 2] += (25 * np.sin(2 * np.pi * 0.11 * t), 12 * np.sin(2 * np.pi * 0.07 * t + 1))
            frame = cv2.warpAffine(frame, m, size, borderMode=cv2.BORDER_REFLECT)
        frame += rng.normal(0, noise, frame.shape).astype(np.float32)
        yield np.clip(frame, 0, 255).astype(np.uint8), t
