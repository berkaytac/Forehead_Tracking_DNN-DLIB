import cv2
import os
import numpy as np
import dlib

class ForeheadTracking:
    def __init__(self):
        # Frame Parameters
        self.frame_in = None
        self.frame_out = None
        self.gray = None
        self.frameWidth = 0
        self.frameHeight = 0
        # Face Detection
        self.modelFile = os.path.join(os.path.abspath("."), "models/res10_300x300_ssd_iter_140000.caffemodel")
        self.configFile = os.path.join(os.path.abspath("."), "models/deploy.prototxt.txt")
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        self.face_width, self.face_height, self.face_center = 0, 0, 0
        self.face_startX, self.face_endX, self.face_endY, self.face_startY = 0, 0, 0, 0
        self.conf = 0
        # Forehead
        self.predictor = dlib.shape_predictor(os.path.join(os.path.abspath("."), "models/shape_predictor_68_face_landmarks.dat"))
        self.forehead_points = None
        self.forehead_startX, self.forehead_startY, self.forehead_endX, self.forehead_endY = 0, 0, 0, 0
        self.forehead_img = None
        # Eye
        self.LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
        self.left_eye_x, self.left_eye_y, self.right_eye_x, self.right_eye_y = 0, 0, 0, 0
        self.eye_angle = 0

    def draw_text(self, frame, text, col, text_pos=(10, 10)):
        """
        Creates a background for the text depending on the text size and puts the text on the image.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 2
        text_color_bg = (0, 0, 0)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(frame, text_pos, (text_pos[0] + text_w, text_pos[1] + text_h), text_color_bg, -1)
        cv2.putText(frame, text, (text_pos[0], text_pos[1] + text_h), font, font_scale, col, font_thickness)
        return frame

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        """
        Get part of the face coordinates based the on the given ratios.
        """
        x, y, w, h = self.face_startX, self.face_startY, self.face_width, self.face_height
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def rotate(self, xy, theta):
        """
        Rotates the point based on the given theta angle
        """
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        return (xy[0] * cos_theta - xy[1] * sin_theta), (xy[0] * sin_theta + xy[1] * cos_theta)

    def translate(self, xy, offset):
        """
        Translates point based on offset
        """
        return xy[0] + offset[0], xy[1] + offset[1]

    def extract_eye_center(self, shape, eye_indices):
        """
        Calculates the eye center by using dlib shape and eye indeces
        """
        points = list(map(lambda i: shape.part(i), eye_indices))
        xs = map(lambda p: p.x, points)
        ys = map(lambda p: p.y, points)
        return sum(xs) // 6, sum(ys) // 6

    def face_shift(self, detected):
        """
        Calculates how much face rectangle is shifted
        """
        x1, y1, x2, y2 = detected
        center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        shift = np.linalg.norm(center - self.face_center)
        self.face_center = center
        return shift

    def dnn_face_detection(self):
        """
        Uses Opencv DNN for the face detection.
        """
        DNN_frame = self.frame_out.copy()
        (self.frameHeight, self.frameWidth) = DNN_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(DNN_frame, (100, 100)), 1.0, (100, 100), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        self.conf = detections[0, 0, 0, 2]
        box = detections[0, 0, 0, 3:7] * np.array(
            [self.frameWidth, self.frameHeight, self.frameWidth, self.frameHeight])
        if self.face_shift(box.astype("int")) > 3:
            self.face_startX, self.face_startY, self.face_endX, self.face_endY = box.astype("int")
            self.face_width = self.face_endX - self.face_startX
            self.face_height = self.face_endY - self.face_startY
        return self.face_startX, self.face_startY, self.face_endX, self.face_endY

    def get_forehead_coord(self):
        """
        This function gets the forehead coordinates based on the face rectangle.
        Then, it calculates the head angle based on the eye centers and applies the rotation to the forehead coordinates
        with perspective transform.
        """
        # Get Forehead coordinates based on Face Rectangle
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.1)
        # Below is for the warping purposes
        shape = self.predictor(self.gray,
                               dlib.rectangle(int(self.face_startX), int(self.face_startY), int(self.face_endX),
                                              int(self.face_endY)))
        # Left and Right Eye
        self.left_eye_x, self.left_eye_y = self.extract_eye_center(shape, self.LEFT_EYE_INDICES)
        self.right_eye_x, self.right_eye_y = self.extract_eye_center(shape, self.RIGHT_EYE_INDICES)
        tan = (self.right_eye_y - self.left_eye_y) / (self.right_eye_x - self.left_eye_x)
        self.eye_angle = np.arctan(tan)
        # top_left, top_right, bottom_right, bottom_left
        points = [(forehead1[0], forehead1[1]),
                  (forehead1[0] + forehead1[2], forehead1[1]),
                  (forehead1[0] + forehead1[2], forehead1[1] + forehead1[3]),
                  (forehead1[0], forehead1[1] + forehead1[3])]
        self.forecentx, self.forecenty = offset = forehead1[0] + (forehead1[2] // 2), forehead1[1] + (forehead1[3] // 2)
        center_offset = (-1 * self.forecentx, -1 * self.forecenty)

        self.forehead_points = np.int0(
            [self.translate(self.rotate(self.translate((xy), center_offset), self.eye_angle), offset) for xy in points])
        self.forehead_startX, self.forehead_startY = self.forehead_points[0][0], self.forehead_points[0][1]
        # warping is done to extract the forehead image.
        dst_pts = np.array(points).astype("float32").reshape(4, 1, 2)
        src_pts = self.forehead_points.astype("float32").reshape(4, 1, 2)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(self.frame_in, M, (self.frameWidth, self.frameHeight))
        self.forehead_img = (
            warped[forehead1[1]:forehead1[1] + forehead1[3], forehead1[0]:forehead1[0] + forehead1[2], :]).astype("int")

    def analyze(self, frame):
        self.frame_in, self.frame_out = frame, frame
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        # Find Face and Draw Face Rectangle.
        self.dnn_face_detection()
        text = "DNN Confidence = " + "{:.0f}%".format(self.conf * 100)
        col = (0, 255, 0)
        self.frame_out = self.draw_text(self.frame_out, text, col)
        cv2.rectangle(self.frame_out, (self.face_startX, self.face_startY),
                      (self.face_startX + self.face_width, self.face_startY + self.face_height), col, 1)
        # Find Eye Centers and Forehead Coordinates with Dlib
        self.get_forehead_coord()
        cv2.drawContours(self.frame_out, [self.forehead_points], 0, col, 2)
        cv2.putText(self.frame_out, "Forehead", (self.forehead_startX, self.forehead_startY), cv2.FONT_HERSHEY_PLAIN, 2,
                    col, 2)
        text = "Forehead Center = {}, {}".format(str(self.forecentx), str(self.forecenty))
        self.frame_out = self.draw_text(self.frame_out, text, (0, 70, 255), (10, 50))
        return self.frame_out
