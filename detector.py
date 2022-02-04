import cv2
import mediapipe as mp
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

        


    def get_pose(self, img, draw = True):
        # #Imported this just to remove certain landmarks
        # from mediapipe.framework.formats import landmark_pb2

        #Convert image from RGB to BGR for vision uses
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.res = self.pose.process(img)
        self.landmarks = self.res.pose_landmarks
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Landmarks
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

        #Only display the shoulders and part of the arms
        # self.landmark_subset = landmark_pb2.NormalizedLandmarkList(
        #     landmark = [
        #         self.landmarks.landmark[11],
        #         self.landmarks.landmark[12],
        #         self.landmarks.landmark[13],
        #         self.landmarks.landmark[14]
        #     ]
        # )


        if self.landmarks:
            if draw:

                #Draws only the shoulders and part of the arms
                #TODO: Figure out how to draw the lines back in without error                
                # self.mp_draw.draw_landmarks(img, 
                #     self.landmark_subset, 
                #     # self.pose_connections,
                
                #     landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())
                self.mp_draw.draw_landmarks(img, 
                self.landmarks, 
                self.pose_connections,
                self.mp_draw.DrawingSpec(color = (0, 255, 255), thickness = 4, circle_radius = 4),
                self.mp_draw.DrawingSpec(color = (0, 0, 255), thickness = 4, circle_radius = 4))
                
                #Backup line
                # self.mp_draw.draw_landmarks(img, 
                #     self.landmarks, 
                #     self.pose_connections,
                #     landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())


        return img

    def calculate_landmarks(self, img):
        #TODO: Understand how to calculate and compare the two shoulders
        pass

    def detect_sitting(self, img):
        #TODO: Create model that could be used to compare sitting position/posture
        pass
    