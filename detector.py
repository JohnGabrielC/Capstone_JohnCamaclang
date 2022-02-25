import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import mediapipe as mp
import numpy as np
import os
import pandas as pd


class Detector:
    def __init__(self, static_mode = False, model_complex = 1, smooth_landmark = True, segmentation = False,
    smooth_segment = False, min_detect_conf = 0.7, min_track_conf = 0.7):
        self.static_mode = static_mode 
        self.model_complex = model_complex
        self.smooth_landmark = smooth_landmark
        self.segmentation = segmentation
        self.smooth_segment = smooth_segment
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf
        self.data = []
        self.pose_tub = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW']
        self.timer = 0

        """
        Args:
        static_mode, treats input image(s) as a video. If set to true, person detection is run through every input image
        model_complex, landmark accuracy and inference latency go up with complexity (0, 1, 2)
        smooth_landmark, true filters landmarks across input to reduce jitter. Ignore if static mode is set ot true
        segmentation, generates a segmentation mask along with the landmarks
        smooth_segment, filters segmentation mask across different input  to reduce jitter. Ignore if segmentation is False or static_moode is True
        min_detect_conf, confidence value from the detection model to be considered successful
        min_tracking_conf, landmark tracking, higher value increases accuracy but could increase latency
        Better definitions found at: https://google.github.io/mediapipe/solutions/pose
        
        """

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(self.static_mode, self.model_complex, self.smooth_landmark,
        self.segmentation, self.smooth_segment, 
        self.min_detect_conf, self.min_track_conf
        )

        
    def get_pose(self, draw = True, win_h = 640, win_w = 480):
        count = 0
        cap = cv2.VideoCapture(0)
        with self.pose:
            while cap.isOpened():
                succ, img = cap.read()
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # img.flags.writeable = False

                res = self.pose.process(img)

                # img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_copy = np.copy(img)
                img = np.zeros(img.shape)

                pose_connections = self.mp_pose.POSE_CONNECTIONS
                landmark_pos = res.pose_landmarks
                landmarks = landmark_pos.landmark

                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                try:

                    cv2.putText(img, str(left_shoulder), 
                        tuple(np.multiply(left_shoulder, [win_h, win_w]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA
                        )
        
                    cv2.putText(img, str(right_shoulder), 
                        tuple(np.multiply(right_shoulder, [win_h, win_w]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA
                        )               
                except:
                    pass
                
                if landmark_pos:
                    if(len(self.data) == 1):
                        pass
                    else:
                        data_tub = {} 

                        for i in range(len(self.pose_tub)):
                            if landmarks[i].x is None or landmarks[i].y is None:
                                landmarks[i].x = None
                                landmarks[i].y = None
                                data_tub.update({self.pose_tub[i]: landmarks[i]})
                            else:
                                landmarks[i].x = landmarks[i].x
                                landmarks[i].y = landmarks[i].y 
                                data_tub.update({self.pose_tub[i]: landmarks[i]})
                            
                        self.data.append(data_tub)


                self.mp_draw.draw_landmarks(img, 
                    landmark_pos, 
                    pose_connections,
                    landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imshow('Pose Outline', img)
                cv2.imshow('', img_copy)
                # self.timer += 1
                if cv2.waitKey(5) & 0xFF == 27:
                    df = pd.DataFrame(self.data)
                    output = "coord.csv"
                    df.to_csv(output, mode = "a", header = not os.path.exists(output))
                    break
                #Check if key is either spacebar or esc
                a = cv2.waitKey(5)

                if a % 256 == 32:
                    self.save_image(count, img)
                    count += 1

            cap.release()
            cv2.destroyAllWindows

    def save_image(self, counter, img):
        path = os.path.dirname(os.path.abspath(__file__)) #Set this to this folder
        cv2.imwrite(os.path.join(path, "frame_%d.jpg" % counter), img)
        print("frame_%d" % counter + " saved")

    
    def detect_sitting(self, img):
        #TODO: Create model that could be used to compare sitting position/posture
        pass


    #TODO: Find walking data for gait analysis

    #TODO: Posture detection and figure out what points are important for detection

    #TODO: importance and impact of this work. What health problems can be detected and can be fixed 

    #TODO scoliosis problem 
