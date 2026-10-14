from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# model files live next to this module, so the app works from any directory
BASE_DIR = Path(__file__).resolve().parent


class Face_utilities():
    '''
    This class contains all needed functions to work with faces in a frame

    Args:
        face_width (int): width/height of the aligned face image
        detect_width (int): frames wider than this are down-scaled before the
            (slow) HOG face detector runs. The landmark predictor still works
            on the full resolution frame, so accuracy is unaffected.
    '''

    def __init__(self, face_width=200, detect_width=320):
        self.detector = None
        self.predictors = {}
        self.age_net = None
        self.gender_net = None

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.gender_list = ['Male', 'Female']

        self.desiredLeftEye = (0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = face_width
        self.detect_width = detect_width

        # For dlib's 68-point facial landmark detector:
        self.FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

        # For dlib's 5-point facial landmark detector. Each eye has two corner
        # points, so these are [start, end) slices covering both of them
        # (the old (2, 3) / (0, 1) only selected a single corner).
        self.FACIAL_LANDMARKS_5_IDXS = OrderedDict([
            ("right_eye", (2, 4)),
            ("left_eye", (0, 2)),
            ("nose", (4, 5))
        ])

        # last params, used by face_full_process
        self.last_age = None
        self.last_gender = None
        self.last_rects = None
        self.last_shape = None
        self.last_aligned_shape = None

    def face_alignment(self, frame, shape):
        '''
        Align the face by vertical axis

        Args:
            frame (cv2 image): the original frame.
            shape (array): 68 (or 5) facial landmarks' co-ords in format of tuples (x,y)

        Outputs:
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): the landmarks in the aligned face
        '''
        n_points = len(shape)
        idxs = self.FACIAL_LANDMARKS_68_IDXS if n_points == 68 else self.FACIAL_LANDMARKS_5_IDXS
        (lStart, lEnd) = idxs["left_eye"]
        (rStart, rEnd) = idxs["right_eye"]

        # center of mass of each eye
        leftEyeCenter = shape[lStart:lEnd].mean(axis=0).astype("int")
        rightEyeCenter = shape[rStart:rEnd].mean(axis=0).astype("int")

        # angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # scale so the eye distance matches the desired one
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0]) * self.desiredFaceWidth
        scale = desiredDist / max(dist, 1e-6)

        # rotation + scaling around the point between the eyes
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # move the eyes to their desired position
        M[0, 2] += self.desiredFaceWidth * 0.5 - eyesCenter[0]
        M[1, 2] += self.desiredFaceHeight * self.desiredLeftEye[1] - eyesCenter[1]

        size = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(frame, M, size, flags=cv2.INTER_CUBIC)

        aligned_shape = cv2.transform(np.reshape(shape, (n_points, 1, 2)), M)
        aligned_shape = np.squeeze(aligned_shape)

        return aligned_face, aligned_shape

    def _get_detector(self):
        if self.detector is None:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
        return self.detector

    def face_detection(self, frame, gray=None):
        '''
        Detect faces in a frame

        Args:
            frame (cv2 image): a normal frame grab from camera or video
            gray (array): optional, the frame already converted to gray scale

        Outputs:
            rects (list): detected faces as dlib rectangles, in the
                coordinates of the full resolution frame
        '''
        if frame is None:
            return

        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = self._get_detector()
        width = gray.shape[1]
        if width <= self.detect_width:
            return detector(gray, 0)

        # the HOG detector cost grows with the pixel count: detect on a small image
        import dlib
        ratio = width / self.detect_width
        small = cv2.resize(gray, None, fx=1 / ratio, fy=1 / ratio, interpolation=cv2.INTER_AREA)
        return [dlib.rectangle(int(r.left() * ratio), int(r.top() * ratio),
                               int(r.right() * ratio), int(r.bottom() * ratio))
                for r in detector(small, 0)]

    def age_gender_detection(self, face):
        '''
        Detect age and gender from a face

        Args:
            face (cv2 image): face after alignment

        Outputs:
            age (str): age
            gender (str): gender
        '''
        if self.age_net is None:
            print("[INFO] load age and gender models ...")
            model_dir = BASE_DIR / "age_gender_models"
            self.age_net = cv2.dnn.readNetFromCaffe(str(model_dir / "deploy_age.prototxt"),
                                                    str(model_dir / "age_net.caffemodel"))
            self.gender_net = cv2.dnn.readNetFromCaffe(str(model_dir / "deploy_gender.prototxt"),
                                                       str(model_dir / "gender_net.caffemodel"))
            print("[INFO] Load models - DONE!")

        if face is None:
            return

        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        gender = self.gender_list[self.gender_net.forward()[0].argmax()]
        self.age_net.setInput(blob)
        age = self.age_list[self.age_net.forward()[0].argmax()]

        return age, gender

    def _get_predictor(self, type):
        if type not in self.predictors:
            import dlib
            print("[INFO] load " + type + " facial landmarks model ...")
            path = BASE_DIR / ("shape_predictor_" + type + "_face_landmarks.dat")
            self.predictors[type] = dlib.shape_predictor(str(path))
            print("[INFO] Load model - DONE!")
        return self.predictors[type]

    def get_landmarks(self, frame, type):
        '''
        Get all facial landmarks in a face

        Args:
            frame (cv2 image): the original frame.
            type (str): 5 or 68 facial landmarks

        Outputs:
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            rects (list): detected faces as rectangles
        '''
        predictor = self._get_predictor(type)

        if frame is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detection(frame, gray)

        if not rects:
            return None, None

        from imutils import face_utils
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        return shape, rects

    def ROI_extraction(self, face, shape):
        '''
        Extract 2 cheeks as the ROIs

        Args:
            face (cv2 image): face cropped from the original frame.
            shape (array): facial landmarks' co-ords in format of tuples (x,y)

        Outputs:
            ROI1 (cv2 image): right-cheek pixels
            ROI2 (cv2 image): left-cheek pixels
            Either can be empty when the landmarks are degenerate.
        '''
        if len(shape) == 68:
            r1 = (shape[29][1], shape[33][1], shape[54][0], shape[12][0])  # right cheek
            r2 = (shape[29][1], shape[33][1], shape[4][0], shape[48][0])   # left cheek
        else:
            top = int((shape[4][1] + shape[2][1]) / 2)
            r1 = (top, shape[4][1], shape[2][0], shape[3][0])
            r2 = (top, shape[4][1], shape[1][0], shape[0][0])

        def crop(box):
            # negative indices would silently wrap around in numpy
            y0, y1, x0, x1 = (max(int(v), 0) for v in box)
            return face[y0:y1, x0:x1]

        return crop(r1), crop(r2)

    def facial_landmarks_remap(self, shape):
        '''
        Convex hull of the landmarks, to be used with cv2.fillConvexPoly
        to build a face mask

        Args:
            shape (array): facial landmarks' co-ords in format of tuples (x,y)

        Outputs:
            hull (array): convex hull of the landmarks
        '''
        return cv2.convexHull(shape)

    def no_age_gender_face_process(self, frame, type):
        '''
        full process to extract face, ROI but no age and gender detection

        Args:
            frame (cv2 image): input frame
            type (str): 5 or 68 landmarks

        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)
            or None if no face was found
        '''
        shape, rects = self.get_landmarks(frame, "5" if type == "5" else "68")
        if shape is None:
            return None

        r = rects[0]
        face = frame[max(r.top(), 0):max(r.bottom(), 0), max(r.left(), 0):max(r.right(), 0)]
        aligned_face, aligned_shape = self.face_alignment(frame, shape)

        return rects, face, shape, aligned_face, aligned_shape

    def face_full_process(self, frame, type, face_detect_on, age_gender_on):
        '''
        full process to extract face, ROI
        face detection and facial landmark run only when face_detect_on is set
        age and gender detection runs only when age_gender_on is set
        last values of detections are used otherwise, to reduce the time of the process

        Args:
            frame (cv2 image): input frame
            type (str): 5 or 68 landmarks
            face_detect_on (bool): flag to run face detection and facial landmarks
            age_gender_on (bool): flag to run age gender detection

        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            (age, gender) (str,str): age and gender
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)
            or None if no face was found
        '''
        age = self.last_age
        gender = self.last_gender
        rects = self.last_rects
        shape = self.last_shape

        if face_detect_on:
            shape, rects = self.get_landmarks(frame, "5" if type == "5" else "68")
            if shape is None:
                return None

        if shape is None or not rects:
            return None

        r = rects[0]
        face = frame[max(r.top(), 0):max(r.bottom(), 0), max(r.left(), 0):max(r.right(), 0)]

        if age_gender_on:
            age, gender = self.age_gender_detection(face)

        aligned_face, aligned_shape = self.face_alignment(frame, shape)

        self.last_age = age
        self.last_gender = gender
        self.last_rects = rects
        self.last_shape = shape
        self.last_aligned_shape = aligned_shape

        return rects, face, (age, gender), shape, aligned_face, aligned_shape
