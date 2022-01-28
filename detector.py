import cv2
import mediapipe as mp

class Detector:
    def __init__(self, static_mode = False, model_complex = 1, smooth_landmark = True, segmentation = False,
    smooth_segment = False, min_detect_conf = 0.6, min_track_conf = 0.6):
        self.static_mode = static_mode
        self.model_complex = model_complex
        self.smooth_landmark = smooth_landmark
        self.segmentation = segmentation
        self.smooth_segment = smooth_segment
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(self.static_mode, self.model_complex, self.smooth_landmark,
        self.segmentation, self.smooth_segment, 
        self.min_detect_conf, self.min_track_conf
        )


    def get_pose(self, img, draw = True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.res = self.pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.res.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img
