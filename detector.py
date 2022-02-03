import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_mp2
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
                self.mp_draw.draw_landmarks(image = img, landmark_list = self.larndmark_subset)
        return img

    def detect_sitting(self, img):
        #TODO: Work on classification model/data
        pass


if __name__ == 'main':
    def main():
        cap = cv2.VideoCapture(0)
        detector = Detector()

        while cap.isOpened():
            succ, img = cap.read()
            if not succ:
                print("Check camera, no output")
                continue
            
            cv2.imshow("Posture Detector", detector.get_pose(img))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        
    main()