import numpy as np

from face_utilities import Face_utilities


def landmarks_68():
    pts = np.zeros((68, 2), int)
    pts[29] = (100, 80)
    pts[33] = (100, 120)
    pts[54], pts[12] = (130, 0), (160, 0)
    pts[4], pts[48] = (40, 0), (70, 0)
    return pts


def test_roi_extraction_68_points():
    face = np.zeros((200, 200, 3), np.uint8)
    right, left = Face_utilities().ROI_extraction(face, landmarks_68())
    assert right.shape[:2] == (40, 30)
    assert left.shape[:2] == (40, 30)


def test_roi_extraction_never_wraps_negative_indices():
    face = np.zeros((200, 200, 3), np.uint8)
    pts = landmarks_68()
    pts[29] = (100, -30)
    pts[33] = (100, 20)
    right, _ = Face_utilities().ROI_extraction(face, pts)
    assert right.shape[0] == 20  # rows 0..20, not a wrapped slice


def test_roi_extraction_degenerate_landmarks_gives_empty_roi():
    face = np.zeros((200, 200, 3), np.uint8)
    pts = landmarks_68()
    pts[33] = (100, 70)  # bottom above top
    right, _ = Face_utilities().ROI_extraction(face, pts)
    assert right.size == 0


def test_alignment_maps_eyes_to_fixed_positions():
    fu = Face_utilities(face_width=200)
    frame = np.zeros((480, 640, 3), np.uint8)
    # dlib 5-point model: points 0,1 = eye on the image right, 2,3 = eye on the image left,
    # 4 = nose. The face is tilted by about 10 degrees.
    shape = np.array([[400, 217], [440, 224], [300, 200], [340, 207], [370, 260]])
    aligned, aligned_shape = fu.face_alignment(frame, shape)
    assert aligned.shape == (200, 200, 3)
    right_eye = aligned_shape[0:2].mean(axis=0)
    left_eye = aligned_shape[2:4].mean(axis=0)
    assert abs(right_eye[1] - left_eye[1]) < 2            # eyes level after alignment
    assert abs((right_eye[0] + left_eye[0]) / 2 - 100) < 3  # centred horizontally
    assert left_eye[0] < right_eye[0]                      # and the face is not flipped
